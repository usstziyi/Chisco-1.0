"""Chisco 句子级 EEG 编码器：TSConv 前端 + 时间注意力池化。

输入是**整句** EEG，不是字级 token：

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
        ↓  Linear
    (B, embedding_dim)
        ↓  L2 归一化

关键设计点：

1. **时间卷积 + 平均池化 + 跨电极空间卷积** —— 与 ``temp/baseline`` 的
   ``TokenEncoder`` 同一套 TSConv 结构，区别只是它作用在整句上而不是单个字 token。
   最后那个 ``kernel=(C, 1)`` 的空间卷积一次性把 122 个电极压成 1，把"通道维"
   彻底消掉，后面就只剩时间维。

2. **不做硬切块、不用 Transformer** —— ``T`` 在 reading / imagine 下分别是
   2501 / 1651，都无法被常见的 patch 长度整除，硬切要补零、还平白多出"patch 长度"
   这个超参。卷积前端本身就带局部时间结构，不需要额外的分块假设。

3. **位置编码必须显式给出** —— 注意力池化对时间维是置换不变的，不注入位置信息
   就分不出"前半句"和"后半句"。句子级 EEG 的时序就是字序，这个信息必须保留。

4. **不需要 mask** —— T 在同一 task 内恒定，batch 里所有样本等长，没有 padding。

5. **GroupNorm 而非 BatchNorm** —— 逐样本归一化，统计量不受 batch 内其他样本影响。

6. **默认不再缩放输入** —— ``ChiscoEEGDataset`` 已经在数据出口把伏特换算成微伏
   （1e-5 V -> 10 µV 量级），所以 ``input_scale`` 默认为 1。这个参数保留着是为了
   兼容"直接用原始伏特值喂模型"的场景，那时设成 1e6 即可：GroupNorm 只保证前向
   尺度不变，首层卷积的梯度会被 1e-6 级别的输入幅度缩掉。

7. **通道维 SE 只是消融开关** —— ``se_mode`` 在四个位置之间切换，全部默认
   ``'none'``，与 baseline 完全一致：

   - ``'front'``：``(B, 1, C, T)`` 前置 SE，对 1 与 T 池化，保留 C=122；
   - ``'back'``：前端输出 ``(B, k, T2)`` 后置 SE，对 T2 池化，保留 k；
   - ``'mid'``：TSConv 内部、空间卷积之前 SE，对 C 与 T2 池化，保留 k。

   三种位置共用同一个 ``SEBlock``，只是 ``dim`` / ``n_chans`` 不同。

用法::

    from se_ablation.model import ConvTimeAttentionEncoder

    model = ConvTimeAttentionEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    eeg = model(batch)          # batch: (B, 122, 2501) -> (B, 768)，已 L2 归一化

    # 消融：se_mode in {'none', 'front', 'back', 'mid'}
    model = ConvTimeAttentionEncoder(n_samples=2501, se_mode="mid")
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """通道维 SE（Squeeze-and-Excitation）门控，输出形状与输入完全一致。

    ``Squeeze`` 对除 batch 维与通道维之外的所有维度做全局平均池化，得到
    ``(B, n_chans)`` 的通道描述子；``Excitation`` 用
    ``Linear -> ReLU -> Linear -> Sigmoid`` 生成逐通道门控，再广播乘回原张量。
    它只重标定通道权重，不改通道数、不改其他维度，因此能插在前向的任意位置。

    支持的输入形状（``dim`` 为通道所在维）::

        (B, C)         dim=1      2D：不池化，直接用 (B, C) 做 Squeeze
        (B, C, T)      dim=1      3D：对 T 池化
        (B, 1, C, T)   dim=2      4D：对 1 与 T 池化（SE_前）
        (B, k, C, T2)  dim=1      4D：对 C 与 T2 池化（SE_折中）

    Parameters
    ----------
    n_chans:
        通道数，必须与输入在 ``dim`` 维上的长度一致。
    reduction:
        压缩比 ``r``，隐层宽度取 ``max(1, n_chans // r)``。
    dim:
        通道维下标，默认 1。作用于 ``(B, 1, C, T)`` 的电极维时传 2。
    init_bias:
        ``fc2.bias`` 的初始值。``sigmoid(0) = 0.5`` 会把特征整体减半、不是恒等
        映射，所以默认取 4.0（门控 ≈ 0.982），让插入 SE 后的初始前向接近原模型。
    """

    def __init__(
        self,
        n_chans: int,
        reduction: int = 4,
        dim: int = 1,
        init_bias: float = 4.0,
    ):
        super().__init__()

        if n_chans <= 0:
            raise ValueError("n_chans 必须为正")
        if reduction <= 0:
            raise ValueError("reduction 必须为正")
        if dim < 1:
            raise ValueError("dim 必须 >= 1（0 是 batch 维，不能当通道维）")

        self.n_chans = n_chans
        self.dim = dim

        hidden = max(1, n_chans // reduction)
        self.fc1 = nn.Linear(n_chans, hidden)
        self.fc2 = nn.Linear(hidden, n_chans)

        # fc2 用极小方差 + 正偏置：初始门控 ≈ 0.982，既不冲掉特征也不完全恒等，
        # 由训练决定到底要不要压掉某些通道
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` 与返回值形状一致，仅在 ``dim`` 维上逐通道缩放。"""
        if x.ndim < 2:
            raise ValueError(f"输入至少要是 (B, C) 二维，收到 {tuple(x.shape)}")
        if self.dim >= x.ndim:
            raise ValueError(f"dim={self.dim} 超出输入维度 {x.ndim}")
        if x.shape[self.dim] != self.n_chans:
            raise ValueError(
                f"期望第 {self.dim} 维为 {self.n_chans}，收到 {x.shape[self.dim]}"
            )

        # Squeeze：只留 batch 与通道维，其余维度全部全局平均掉
        pool_dims = tuple(d for d in range(x.ndim) if d not in (0, self.dim))
        desc = x.mean(dim=pool_dims) if pool_dims else x  # (B, n_chans)

        # Excitation：通道间依赖 -> (0, 1) 门控
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(desc))))  # (B, n_chans)

        # 广播回原形状：(B, C) -> (B, 1, ..., n_chans, ..., 1)
        shape = [1] * x.ndim
        shape[0] = x.shape[0]
        shape[self.dim] = self.n_chans
        return x * gate.reshape(shape)


class TSConvFrontend(nn.Module):
    """把整句 ``(B, 1, C, T)`` 压成时间序列 ``(B, k, T2)``。

    ``T2`` 取决于 ``T``；输入长度决定了一切，所以 ``out_time`` 是构造模型时
    确定位置编码长度的依据。

    为了能插入 SE_折中，原来那个整块的 ``nn.Sequential`` 拆成了两段：

    - ``conv_time``：时间卷积 + 平均池化 + GroupNorm + ELU，(B,1,C,T) -> (B,k,C,T2)
    - ``conv_space``：空间卷积（跨全部电极） + GroupNorm + ELU + Dropout，
      (B,k,C,T2) -> (B,k,1,T2)
    - 两段之间的 ``se`` 在 ``se_mode='mid'`` 时才存在。

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 122（仅 EEG，已剔除 VEO / HEO / Trigger）。
    k:
        时间卷积的输出通道数，也是后面 Transformer 之外的"特征宽度"。
    m1:
        时间卷积核长度（采样点）。500 Hz 下取 25 ≈ 50 ms。
    m2 / s:
        时间平均池化的核长与步长，共同决定时间维的下采样倍率 ``s``。
    drop_prob:
        空间卷积之后的 dropout 概率。
    se_mode:
        ``'none'`` 不插 SE；``'mid'`` 在时间卷积与空间卷积之间插 ``SEBlock``。
    se_reduction:
        SE 的压缩比，透传给 ``SEBlock``。
    """

    def __init__(
        self,
        n_chans: int = 122,
        k: int = 32,
        m1: int = 25,
        m2: int = 25,
        s: int = 5,
        drop_prob: float = 0.3,
        se_mode: str = "none",
        se_reduction: int = 4,
    ):
        super().__init__()

        if n_chans <= 0 or k <= 0:
            raise ValueError("n_chans / k 必须为正")
        if m1 <= 0 or m2 <= 0 or s <= 0:
            raise ValueError("m1 / m2 / s 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")
        if se_mode not in ("none", "mid"):
            raise ValueError(
                f"TSConvFrontend 只支持 se_mode='none'/'mid'，收到 {se_mode!r}"
            )

        self.n_chans = n_chans
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.s = s
        self.se_mode = se_mode

        # 时间段：时间卷积 -> 平均池化 -> 归一化 -> 激活，(B,1,C,T) -> (B,k,C,T2)
        self.conv_time = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=k, kernel_size=(1, m1)),
            nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s)),
            nn.GroupNorm(num_groups=1, num_channels=k),
            nn.ELU(),
        )

        # SE_折中：插在空间卷积之前，对 C 与 T2 池化，保留 k 个特征通道
        self.se = (
            SEBlock(k, reduction=se_reduction, dim=1) if se_mode == "mid" else None
        )

        # 空间段：一次跨越全部电极 -> 归一化 -> 激活 -> dropout，(B,k,C,T2) -> (B,k,1,T2)
        self.conv_space = nn.Sequential(
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

        x = self.conv_time(x)  # (B, k, C, T2)
        if self.se is not None:
            # SE_折中：(B, k, C, T2) 上对 C 与 T2 池化，保留 k
            x = self.se(x)
        # (B, k, 1, T2) -> (B, k, T2)：第 2 维已被空间卷积压成 1
        return self.conv_space(x).squeeze(2)


class ConvTimeAttentionEncoder(nn.Module):
    """句子级 EEG 编码器，输出与文本共享空间的定长 embedding。

    Parameters
    ----------
    n_samples:
        输入时间长度 ``T``，必须与数据集给出的形状一致
        （``ChiscoEEGDataset`` 下 reading = 2501、imagine = 1651，500 Hz）。
    n_chans:
        EEG 通道数，本数据集为 122。
    k / m1 / m2 / s / drop_prob:
        透传给 ``TSConvFrontend``（特征宽度 / 时间卷积核 / 池化核 / 池化步长 /
        dropout），含义见 ``TSConvFrontend``。
    d_model / n_heads / dropout:
        注意力池化的配置。时间步最多几百个、单被试数据量约 6000，
        默认给得很浅（单层注意力池化，不加 Transformer）。
    embedding_dim:
        最终 embedding 维度；与文本侧对齐，LaBSE 是 768。
    input_scale:
        输入放大系数，默认 1（``ChiscoEEGDataset`` 已返回微伏），见模块文档第 6 点。
    se_mode:
        通道维 SE 的插入位置，四个取值构成一次消融实验，默认 ``'none'``（等于
        baseline）：

        - ``'none'``：不插 SE；
        - ``'front'``：``(B, 1, C, T)`` 前置 SE，对 1 与 T 池化，保留 C=122；
        - ``'back'``：前端输出 ``(B, k, T2)`` 后置 SE，对 T2 池化，保留 k；
        - ``'mid'``：透传给 ``TSConvFrontend``，在空间卷积之前 SE，对 C 与 T2
          池化，保留 k。
    se_reduction:
        SE 的压缩比，透传给所有 ``SEBlock``。

    Attributes
    ----------
    n_time:
        TSConv 前端输出的时间步数 ``T2``，也是位置编码的长度。
    """

    def __init__(
        self,
        n_samples: int,
        n_chans: int = 122,
        k: int = 32,
        m1: int = 25,
        m2: int = 25,
        s: int = 5,
        drop_prob: float = 0.3,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        embedding_dim: int = 768,
        input_scale: float = 1.0,
        se_mode: str = "none",
        se_reduction: int = 4,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if input_scale <= 0:
            raise ValueError("input_scale 必须为正")
        if se_mode not in ("none", "front", "back", "mid"):
            raise ValueError(
                f"se_mode 只能是 'none'/'front'/'back'/'mid'，收到 {se_mode!r}"
            )

        self.n_samples = n_samples
        self.n_chans = n_chans
        self.k = k
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.input_scale = input_scale
        self.se_mode = se_mode

        # SE_折中 在 TSConvFrontend 内部，所以把开关透传下去
        self.frontend = TSConvFrontend(
            n_chans=n_chans,
            k=k,
            m1=m1,
            m2=m2,
            s=s,
            drop_prob=drop_prob,
            se_mode="mid" if se_mode == "mid" else "none",
            se_reduction=se_reduction,
        )
        self.n_time = self.frontend.out_time(n_samples)

        # SE_前：作用在原始电极维，(B, 1, C, T) 上对 1 与 T 池化，保留 C=122
        self.se_front = (
            SEBlock(n_chans, reduction=se_reduction, dim=2)
            if se_mode == "front"
            else None
        )

        # SE_后：作用在前端输出的 k 个特征通道，(B, k, T2) 上对 T2 池化，保留 k
        self.se_back = (
            SEBlock(k, reduction=se_reduction, dim=1) if se_mode == "back" else None
        )

        self.proj = nn.Linear(k, d_model)

        # 句子的时间顺序就是字序，池化本身对顺序不敏感，位置信息必须显式给出
        self.position = nn.Parameter(torch.zeros(1, self.n_time, d_model))

        # attention pooling：一个可学习 query 对全部时间步做加权汇总，
        # 唯一目的就是把变长时间序列池化成一个定长向量。
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 只有这两个是手工建的 torch.zeros Parameter，必须自己打散；
        # 其余模块（含 out_proj / proj 的 nn.Linear）交给 PyTorch 默认初始化即可。
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

        x = eeg.unsqueeze(1) * self.input_scale  # (B, 1, C, T)
        if self.se_front is not None:
            # SE_前：(B, 1, C, T) 上对 1 与 T 池化，保留 C
            x = self.se_front(x)

        hidden = self.frontend(x)                # (B, k, T2)
        if self.se_back is not None:
            # SE_后：(B, k, T2) 上对 T2 池化，保留 k
            hidden = self.se_back(hidden)

        seq = self.proj(hidden.transpose(1, 2))  # (B, T2, d_model)
        seq = seq + self.position

        query = self.pool_query.expand(seq.size(0), -1, -1)
        pooled, _ = self.pool_attn(query, seq, seq)  # (B, 1, d_model)

        return F.normalize(self.out_proj(pooled.squeeze(1)), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = ConvTimeAttentionEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total:,}（可训练 {trainable:,}）")
    print(f"T2 时间步={model.n_time}（输入 T={model.n_samples}）")

    batch = torch.randn(4, 122, 2501) * 10.0  # 微伏量级，与 ChiscoEEGDataset 的输出一致
    model.eval()
    with torch.no_grad():
        eeg = model(batch)

    print(f"\n输入={tuple(batch.shape)} -> 输出={tuple(eeg.shape)}")
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # imagine 的 T 不同，必须重建模型
    imagine = ConvTimeAttentionEncoder(n_samples=1651, n_chans=122, embedding_dim=768)
    print(f"\nimagine 模型 T2 时间步={imagine.n_time}")

    # ---------------- 通道维 SE 消融：四种 se_mode 的形状与增量参数量 ----------------
    print("\nSE 消融（k=32, C=122, reduction=4）:")
    for mode in ("none", "front", "back", "mid"):
        net = ConvTimeAttentionEncoder(
            n_samples=2501, n_chans=122, embedding_dim=768, se_mode=mode
        )
        n_params = sum(p.numel() for p in net.parameters())
        net.eval()
        with torch.no_grad():
            out = net(batch)

        # 初始门控：喂全零描述子，验证 fc2 正偏置让门控从 ≈0.98 起步
        se = {"front": net.se_front, "back": net.se_back, "mid": net.frontend.se}
        block = se.get(mode)
        gate_mean = (
            torch.sigmoid(block.fc2(torch.relu(block.fc1(torch.zeros(1, block.n_chans)))))
            .mean()
            .item()
            if block is not None
            else float("nan")
        )
        print(
            f"  se_mode={mode:<5} 参数量={n_params:,} (+{n_params - total:,}) "
            f"输出={tuple(out.shape)} 初始门控均值={gate_mean:.4f}"
        )
