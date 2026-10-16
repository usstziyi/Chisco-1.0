"""Chisco 句子级 EEG 分类器：TSConv 前端 + 时间注意力池化 + 线性分类头。

本实验**只编码 EEG**，文本侧完全去掉（对比 ``baseline/model.py``：那里输出要与
LaBSE 文本 embedding 对齐，所以末尾有一个 768 维投影 + L2 归一化）。这里标签是
句子语义类别 ``type_number``（0~38），是个 39 类分类问题，所以末尾换成分类头，
输出 logits：

    (B, C, T)                  # C = 122（仅 EEG 通道），T 由 task 决定、整表恒定
        ↓  ×input_scale，再 unsqueeze 成 (B, 1, C, T)
    TSConv 前端（权重共享，逐样本编码整句）
        ↓
    (B, k, T2)
        ↓  transpose + Linear(k → d_model)
    (B, T2, d_model)
        ↓  + 可学习位置编码
        ↓  可学习 query 对 T2 个时间步做 attention pooling
    (B, d_model)
        ↓  Dropout + Linear(d_model → num_classes)
    (B, num_classes)           # 未过 softmax，直接喂 CrossEntropyLoss

关键设计点（与方案 A 同源的部分）：

1. **时间卷积 + 平均池化 + 跨电极空间卷积** —— 与 ``baseline/model.py`` 的
   ``TSConvFrontend`` 逐字相同的结构：最后那个 ``kernel=(C, 1)`` 的空间卷积一次性
   把 122 个电极压成 1，把"通道维"彻底消掉，后面就只剩时间维。
2. **不做硬切块、不用 Transformer** —— ``T`` 在 reading / imagine 下分别是
   2501 / 1651，都无法被常见的 patch 长度整除；卷积前端本身就带局部时间结构。
3. **位置编码必须显式给出** —— 注意力池化对时间维是置换不变的，不注入位置信息
   就分不出"前半句"和"后半句"。
4. **不需要 mask** —— T 在同一 task 内恒定，batch 里所有样本等长，没有 padding。
5. **GroupNorm 而非 BatchNorm** —— 逐样本归一化，统计量不受 batch 内其他样本影响。
6. **默认不再缩放输入** —— ``ChiscoEEGClassificationDataset`` 已在数据出口把伏特
   换算成微伏，所以 ``input_scale`` 默认为 1，见 ``baseline/model.py`` 第 6 点。

相比方案的差异：池化后的向量**不做 L2 归一化**（归一化只服务于"点积即余弦"的
检索目标，对分类没有意义，反而丢掉了模长这一可用的置信度信息），而是直接送进
分类头。``encode()`` 单独暴露池化向量，方便做线性探针 / 特征分析。

用法::

    from eeg_classification.model import EEGClassifier

    model = EEGClassifier(n_samples=2501, n_classes=39, n_chans=122)
    logits = model(batch)       # batch: (B, 122, 2501) -> (B, 39)
    feats = model.encode(batch) # (B, d_model)，池化后的整句表征
"""

from __future__ import annotations

import torch
from torch import nn


class TSConvFrontend(nn.Module):
    """把整句 ``(B, 1, C, T)`` 压成时间序列 ``(B, k, T2)``（同 ``baseline.model``）。

    ``T2`` 取决于 ``T``；输入长度决定了一切，所以 ``out_time`` 是构造模型时
    确定位置编码长度的依据。

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 122（仅 EEG，已剔除 VEO / HEO / Trigger）。
    k:
        时间卷积的输出通道数，也是后面的"特征宽度"。
    m1:
        时间卷积核长度（采样点）。500 Hz 下取 25 ≈ 50 ms。
    m2 / s:
        时间平均池化的核长与步长，共同决定时间维的下采样倍率 ``s``。
    drop_prob:
        空间卷积之后的 dropout 概率。
    """

    def __init__(
        self,
        n_chans: int = 122,
        k: int = 32,
        m1: int = 25,
        m2: int = 25,
        s: int = 5,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or k <= 0:
            raise ValueError("n_chans / k 必须为正")
        if m1 <= 0 or m2 <= 0 or s <= 0:
            raise ValueError("m1 / m2 / s 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        self.n_chans = n_chans
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.s = s

        self.conv = nn.Sequential(
            # 时间卷积：(B, 1, C, T) -> (B, k, C, T1)
            nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, m1)),
            # 时间平均池化：(B, k, C, T1) -> (B, k, C, T2)
            nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            # 空间卷积：一次跨越全部电极，(B, k, C, T2) -> (B, k, 1, T2)
            nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            nn.Dropout(drop_prob),
        )

    def out_time(self, n_samples: int) -> int:
        """给定输入时间长度 ``T``，算出输出时间长度 ``T2``。"""
        if n_samples <= 0:
            raise ValueError("n_samples 必须为正")
        t1 = n_samples - self.m1 + 1
        if t1 < self.m2:
            raise ValueError(
                f"n_samples={n_samples} 太短：m1={self.m1}, m2={self.m2} 至少需要 "
                f"{self.m1 + self.m2 - 1} 个采样点"
            )
        return (t1 - self.m2) // self.s + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, C, T)`` -> ``(B, k, T2)``。"""
        if x.ndim != 4:
            raise ValueError(f"输入必须是 (B, 1, C, T) 四维，收到 {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"期望第 1 维为 1，收到 {x.shape[1]}")
        if x.shape[2] != self.n_chans:
            raise ValueError(f"期望 {self.n_chans} 个 EEG 通道，收到 {x.shape[2]}")

        # (B, k, 1, T2) -> (B, k, T2)：第 2 维已被空间卷积压成 1
        return self.conv(x).squeeze(2)


class EEGClassifier(nn.Module):
    """句子级 EEG 分类器：``(B, C, T)`` -> ``(B, num_classes)`` logits。

    Parameters
    ----------
    n_samples:
        输入时间长度 ``T``，必须与数据集给出的形状一致
        （reading = 2501、imagine = 1651，500 Hz）。
    n_classes:
        类别数，本实验为 39（``type_number`` 0~38）。
    n_chans:
        EEG 通道数，本数据集为 122。
    k / m1 / m2 / s / drop_prob:
        透传给 ``TSConvFrontend`` 前端。
    d_model / n_heads / dropout:
        注意力池化的配置。时间步最多几百个、单被试数据量约 6000，
        默认给得很浅（单层注意力池化，不加 Transformer）。
    head_dropout:
        分类头之前的 dropout 概率；整句只有一个样本、类别又细，dropout 是主要的
        正则手段。
    input_scale:
        输入放大系数，默认 1（数据集已返回微伏），见模块文档第 6 点。

    Attributes
    ----------
    n_time:
        TSConv 前端输出的时间步数 ``T2``，也是位置编码的长度。
    """

    def __init__(
        self,
        n_samples: int,
        n_classes: int,
        n_chans: int = 122,
        k: int = 32,
        m1: int = 25,
        m2: int = 25,
        s: int = 5,
        drop_prob: float = 0.3,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        input_scale: float = 1.0,
    ):
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes 至少为 2，收到 {n_classes}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if input_scale <= 0:
            raise ValueError("input_scale 必须为正")

        self.n_samples = n_samples
        self.n_chans = n_chans
        self.n_classes = n_classes
        self.k = k
        self.d_model = d_model
        self.input_scale = input_scale

        self.frontend = TSConvFrontend(
            n_chans=n_chans, k=k, m1=m1, m2=m2, s=s, drop_prob=drop_prob
        )
        self.n_time = self.frontend.out_time(n_samples)

        self.proj = nn.Linear(k, d_model)

        # 句子的时间顺序就是字序，池化本身对顺序不敏感，位置信息必须显式给出
        self.position = nn.Parameter(torch.zeros(1, self.n_time, d_model))

        # attention pooling：一个可学习 query 对全部时间步做加权汇总，
        # 唯一目的就是把变长时间序列池化成一个定长向量。
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, n_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 只有这两个是手工建的 torch.zeros Parameter，必须自己打散；
        # 其余模块（含 proj / head 的 nn.Linear）交给 PyTorch 默认初始化即可。
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def _check_input(self, eeg: torch.Tensor) -> None:
        if eeg.ndim != 3:
            raise ValueError(f"eeg 必须是 (B, C, T) 三维，收到 {tuple(eeg.shape)}")
        if eeg.shape[1] != self.n_chans:
            raise ValueError(f"期望 {self.n_chans} 个 EEG 通道，收到 {eeg.shape[1]}")
        if eeg.shape[2] != self.n_samples:
            raise ValueError(
                f"期望时间长度 {self.n_samples}，收到 {eeg.shape[2]}；"
                "reading / imagine 的 T 不同，换 task 时要重建模型"
            )

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """池化后的整句表征 ``(B, d_model)``（未归一化），可用于线性探针。"""
        self._check_input(eeg)

        x = eeg.unsqueeze(1) * self.input_scale  # (B, 1, C, T)
        hidden = self.frontend(x)                # (B, k, T2)

        seq = self.proj(hidden.transpose(1, 2))  # (B, T2, d_model)
        seq = seq + self.position

        query = self.pool_query.expand(seq.size(0), -1, -1)
        pooled, _ = self.pool_attn(query, seq, seq)  # (B, 1, d_model)
        return pooled.squeeze(1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg:
            ``(B, C, T)``，单位微伏（``ChiscoEEGClassificationDataset`` 的输出）。

        Returns
        -------
        torch.Tensor:
            ``(B, n_classes)`` logits，**未经 softmax**；直接喂 ``CrossEntropyLoss``。
        """
        return self.head(self.encode(eeg))


if __name__ == "__main__":
    torch.manual_seed(0)

    model = EEGClassifier(n_samples=2501, n_classes=39, n_chans=122)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(f"T2 时间步={model.n_time}（输入 T={model.n_samples}）")

    batch = torch.randn(4, 122, 2501) * 10.0  # 微伏量级，与 dataset 的输出一致
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        feats = model.encode(batch)

    print(f"\n输入={tuple(batch.shape)} -> logits={tuple(logits.shape)} 表征={tuple(feats.shape)}")
    print("logits 每行和（未归一化，应不为 1）:", logits.sum(dim=-1).tolist())

    # imagine 的 T 不同，必须重建模型
    imagine = EEGClassifier(n_samples=1651, n_classes=39, n_chans=122)
    print(f"\nimagine 模型 T2 时间步={imagine.n_time}")
