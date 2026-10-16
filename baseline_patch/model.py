"""方案 B（对照实验）：时间分块 + 共享 CNN + Transformer + 注意力池化。

这是 ``baseline/model.py``（方案 A）的对照：同样是单被试、整句 EEG ``(B, C, T)``
输入、同样输出 L2 归一化的 ``(B, 768)``，唯一的差别是**在时间轴上引入了硬切的
patch 结构**，并把 Transformer 当成序列模型来用。

    (B, C, T)
        ↓  ×input_scale，并丢掉不足一个 patch 的尾巴（T -> n_patches × patch_samples）
    reshape 成 (B, n_patches, C, patch_samples)
        ↓  共享权重逐 patch 编码（patch tokenizer）
    (B, n_patches, d_model)
        ↓  + 可学习位置编码
    TransformerEncoder（n_layers 层）
        ↓  可学习 query 做 attention pooling
    (B, d_model)
        ↓  Linear -> embedding_dim
    (B, embedding_dim)  -> L2 归一化

与方案 A 的差异，以及为什么它只是对照：

1. **patch 边界是人为引入的超参** —— A 没有任何切分假设，卷积前端直接在整句上
   工作；B 必须选一个 ``patch_samples``，这个选择本身没有数据层面的依据。
2. **T 无法被 patch 长度整除** —— 本数据集 reading ``T=2501``、imagine ``T=1651``，
   都是质数结构，常见 patch 长度都除不尽。这里既不补零（补零会造出一个几乎全是
   填充的垃圾 patch，还要额外处理 mask），也不做重叠切分，而是**直接丢掉不足一个
   patch 的尾巴**：``patch_samples=50`` 时两个 task 都只丢 1 个采样点（0.2 s 的
   2% 以下），代价可以忽略。
3. **仍然不需要 mask** —— T 在同一 task 内恒定，丢掉尾巴之后所有样本的 patch 数
   完全相同，batch 里没有长度差异。
4. **Transformer 在小数据上更容易过拟合** —— 单被试约 6000 条样本，默认只给
   2 层、4 头的浅配置。

用法::

    from baseline_patch.model import PatchTransformerEncoder

    model = PatchTransformerEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    eeg = model(batch)          # batch: (B, 122, 2501) -> (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PatchTokenEncoder(nn.Module):
    """把一个时间 patch ``(C, patch_samples)`` 编码成 ``d_model`` 维向量。

    NICE 风格的 TSConv 小 CNN：先在时间轴上做卷积 + 平均池化，再用一个跨越全部
    电极的空间卷积把通道维压掉。权重在所有 patch 之间共享（相当于 ViT 的 patch
    embedding），否则 patch 数一变参数就不够用。

    注意：归一化用 ``GroupNorm(1, k)`` 而非 ``BatchNorm2d`` —— 逐样本归一化，
    统计量不受 batch 内其他样本影响。

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 122。
    patch_samples:
        每个 patch 的采样点数，本文件默认 50（500 Hz 下 100 ms）。
    """

    def __init__(
        self,
        n_chans: int = 122,
        patch_samples: int = 50,
        d_model: int = 128,
        k: int = 32,
        m1: int = 9,
        m2: int = 13,
        s: int = 2,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or patch_samples <= 0 or d_model <= 0 or k <= 0:
            raise ValueError("n_chans / patch_samples / d_model / k 必须为正")
        if m1 <= 0 or m2 <= 0 or s <= 0:
            raise ValueError("m1 / m2 / s 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        t1 = patch_samples - m1 + 1
        if t1 < m2:
            raise ValueError(
                f"patch_samples={patch_samples} 太短：m1={m1}, m2={m2} 至少需要 "
                f"{m1 + m2 - 1} 个采样点"
            )
        t2 = (t1 - m2) // s + 1

        self.n_chans = n_chans
        self.patch_samples = patch_samples
        self.d_model = d_model
        self.out_time = t2
        self.flatten_dim = k * t2

        self.tsconv = nn.Sequential(
            # 时间卷积：(B·p, 1, C, P) -> (B·p, k, C, T1)
            nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, m1)),
            # 时间平均池化：(B·p, k, C, T1) -> (B·p, k, C, T2)
            nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            # 空间卷积：一次跨越全部电极，(B·p, k, C, T2) -> (B·p, k, 1, T2)
            nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
            nn.Dropout(drop_prob),
        )
        self.proj = nn.Linear(self.flatten_dim, d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: ``(N, C, patch_samples)`` -> ``(N, d_model)``。"""
        if patches.ndim != 3:
            raise ValueError(
                f"patch 必须是 (N, C, P) 三维，收到 {tuple(patches.shape)}"
            )
        if patches.shape[1] != self.n_chans:
            raise ValueError(
                f"期望 {self.n_chans} 个 EEG 通道，收到 {patches.shape[1]}"
            )
        if patches.shape[2] != self.patch_samples:
            raise ValueError(
                f"期望每个 patch {self.patch_samples} 个采样点，收到 {patches.shape[2]}"
            )

        x = self.tsconv(patches.unsqueeze(1))  # (N, 1, C, P) -> (N, k, 1, T2)
        return self.proj(x.flatten(start_dim=1))  # (N, d_model)


class PatchTransformerEncoder(nn.Module):
    """句子级 EEG 编码器（方案 B 对照），输出与文本共享空间的定长 embedding。

    Parameters
    ----------
    n_samples:
        输入时间长度 ``T``，必须与数据集给出的形状一致
        （``ChiscoEEGDataset`` 下 reading = 2501、imagine = 1651，500 Hz）。
    n_chans:
        EEG 通道数，本数据集为 122。
    patch_samples:
        每个 patch 的采样点数；``n_patches = n_samples // patch_samples``，
        不足一个 patch 的尾巴被丢弃。
    d_model / n_heads / n_layers / dim_feedforward / dropout:
        Transformer 配置。token 数只有几十个、数据量也小，默认给得很浅
        （2 层、4 头）。
    embedding_dim:
        最终 embedding 维度；与文本侧对齐，LaBSE 是 768。
    input_scale:
        输入放大系数，默认 1（``ChiscoEEGDataset`` 已返回微伏）；只有直接用原始
        伏特值喂模型时才需要设成 1e6，原因同方案 A。
    token_kwargs:
        透传给 ``PatchTokenEncoder``（k / m1 / m2 / s / drop_prob）。

    Attributes
    ----------
    n_patches:
        patch 数，也是位置编码的长度。
    dropped_samples:
        被丢掉的尾巴采样点数 ``T - n_patches * patch_samples``。
    """

    def __init__(
        self,
        n_samples: int,
        n_chans: int = 122,
        patch_samples: int = 50,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        embedding_dim: int = 768,
        input_scale: float = 1.0,
        **token_kwargs,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if input_scale <= 0:
            raise ValueError("input_scale 必须为正")

        self.n_patches = n_samples // patch_samples
        if self.n_patches < 1:
            raise ValueError(
                f"n_samples={n_samples} 不足一个 patch（patch_samples={patch_samples}）"
            )

        self.n_samples = n_samples
        self.n_chans = n_chans
        self.patch_samples = patch_samples
        self.dropped_samples = n_samples - self.n_patches * patch_samples
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.input_scale = input_scale

        self.patch_encoder = PatchTokenEncoder(
            n_chans=n_chans,
            patch_samples=patch_samples,
            d_model=d_model,
            **token_kwargs,
        )

        # patch 的时间顺序就是字序，必须显式给出位置信息
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True 时 PyTorch 无法启用 nested tensor 加速，显式关掉以免告警
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # attention pooling：一个可学习 query 对全部 patch 做加权汇总
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 只有这两个是手工建的 torch.zeros Parameter，必须自己打散；
        # 其余模块（含 out_proj 的 nn.Linear）交给 PyTorch 默认初始化即可。
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg:
            ``(B, C, T)``；取值单位沿用 MNE 约定（伏特）。

        Returns
        -------
        torch.Tensor:
            ``(B, embedding_dim)``，已做 L2 归一化；与文本 embedding 点积即余弦相似度。
        """
        if eeg.ndim != 3:
            raise ValueError(f"eeg 必须是 (B, C, T) 三维，收到 {tuple(eeg.shape)}")
        if eeg.shape[1] != self.n_chans:
            raise ValueError(f"期望 {self.n_chans} 个 EEG 通道，收到 {eeg.shape[1]}")
        if eeg.shape[2] != self.n_samples:
            raise ValueError(
                f"期望时间长度 {self.n_samples}，收到 {eeg.shape[2]}；"
                "reading / imagine 的 T 不同，换 task 时要重建模型"
            )

        batch_size = eeg.size(0)
        usable = self.n_patches * self.patch_samples
        # 丢掉不足一个 patch 的尾巴
        x = eeg[:, :, :usable] * self.input_scale  # (B, C, usable)

        # (B, C, usable) -> (B, n_patches, C, patch_samples)
        patches = x.reshape(
            batch_size, self.n_chans, self.n_patches, self.patch_samples
        ).transpose(1, 2)

        # 所有 patch 共用同一套权重
        flat = patches.reshape(
            batch_size * self.n_patches, self.n_chans, self.patch_samples
        )
        hidden = self.patch_encoder(flat).reshape(
            batch_size, self.n_patches, self.d_model
        )
        hidden = hidden + self.position
        hidden = self.transformer(hidden)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attn(query, hidden, hidden)  # (B, 1, d_model)

        return F.normalize(self.out_proj(pooled.squeeze(1)), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    for n_samples in (2501, 1651):
        model = PatchTransformerEncoder(
            n_samples=n_samples, n_chans=122, embedding_dim=768
        )
        total = sum(p.numel() for p in model.parameters())
        print(
            f"T={n_samples} -> patch 数={model.n_patches} "
            f"丢弃尾巴={model.dropped_samples} 采样点 | 参数量={total:,}"
        )

    model = PatchTransformerEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    print(f"\npatch encoder 输出时间长度={model.patch_encoder.out_time} "
          f"flatten={model.patch_encoder.flatten_dim}")

    batch = torch.randn(4, 122, 2501) * 10.0  # 微伏量级，与 ChiscoEEGDataset 的输出一致
    model.eval()
    with torch.no_grad():
        eeg = model(batch)

    print(f"输入={tuple(batch.shape)} -> 输出={tuple(eeg.shape)}")
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # 尾巴被丢弃，改动尾部采样点不应该影响输出
    altered = batch.clone()
    altered[:, :, model.n_patches * model.patch_samples :] = torch.randn_like(
        altered[:, :, model.n_patches * model.patch_samples :]
    )
    with torch.no_grad():
        eeg_altered = model(altered)
    print("改动尾巴后的最大输出偏差:", (eeg - eeg_altered).abs().max().item())
