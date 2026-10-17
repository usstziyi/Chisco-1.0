"""方案 A'（对照实验）：多尺度时间卷积的句子级 EEG 编码器。

这是 ``baseline/model.py``（方案 A）的对照：输入、输出、损失与评测完全一致，
**唯一改动是 TSConv 前端的时间感受野**——由单一 ``Conv(1×25)`` 换成三条并行
时间卷积，再配一个更短的池化窗。

    (B, C, T)
        ↓  ×input_scale，并 unsqueeze 成 (B, 1, C, T)
    多尺度时间卷积（三支并行，same padding，边界 replicate）
        Conv(1→11, 1×9)   ┐
        Conv(1→11, 1×25)  ├→ concat → (B, 32, C, T)
        Conv(1→10, 1×51)  ┘
        ↓  AvgPool(1×9, s=5)
    (B, k, T2)
        ↓  transpose + Linear(k → d_model)
    (B, T2, d_model)
        ↓  + 可学习位置编码
        ↓  可学习 query 对 T2 个时间步做 attention pooling
    (B, d_model)
        ↓  Linear
    (B, embedding_dim)  -> L2 归一化

尺度划分（500 Hz，1 采样点 = 2 ms）
-----------------------------------
三支核宽按约 2 倍的几何级数铺开，覆盖三个相邻频段，且中间那支与方案 A 的
``m1=25`` 完全一致，保证对照组里保留一条"原配置"：

    分支    核宽    时长      一阶零点（500/m）    主要覆盖
    快       9      18 ms     ≈ 55.6 Hz           gamma / 高 beta
    中      25      50 ms     ≈ 20.0 Hz           beta / alpha
    慢      51     102 ms     ≈  9.8 Hz           alpha / theta / 慢漂移

32 个特征图按 ``11 / 11 / 10`` 分给三支，参数为 ``11·9 + 11·25 + 10·51 = 884``，
与方案 A 的 ``32·25 = 800`` 同量级，因此对比不会被时间前端的参数量差异带偏。

池化为什么要跟着改
------------------
方案 A 的 ``AvgPool(1×25, s=5)`` 是 50 ms 的滑动平均，一阶零点恰好也在 20 Hz，
等于把 20 Hz 以上全部压掉——快支刚带进来的 gamma / 高 beta 会被这一步重新平均没了，
多尺度就退化成单一低通。这里改成 ``AvgPool(1×9, s=5)``：18 ms 窗、10 ms 跳步，
一阶零点约 55.6 Hz，与快支的核宽对齐，高频得以保留到 attention pooling 之前。

边界为什么用 replicate
----------------------
三支卷积都是 same padding，最慢一支在句子首尾各要补 25 个点（50 ms）。默认的
零填充会在边界处制造一个从 0 跳到 EEG 幅度的阶跃，卷积后变成首尾的伪瞬变，
而慢支核最宽，正好最容易被这个阶跃影响。``padding_mode="replicate"`` 用边界
采样值向外复制，不引入零点，波形在边界处保持连续；代价是首尾若干时间步近似
"重复最后一帧"，但它们只占整句的 2~3%，且会被 attention pooling 按权重吸收。

replicate 不改变输出长度与参数量：padding 仍是对称的 ``m//2``，卷积输出仍是 T。
PyTorch 要求 replicate 的 padding 小于输入长度，这里最大 padding 为 25、
T ≥ 1651，始终满足。

停步仍是 5（10 ms），所以输出长度几乎不变：

    reading  T=2501 -> T2=499（方案 A 为 491）
    imagine  T=1651 -> T2=329（方案 A 为 321）

位置编码与 attention pooling 的长度、显存和计算量都与方案 A 可比，指标差异可以
归因到时间前端，而不是序列长度。

其余部分——GroupNorm、跨电极空间卷积、位置编码、attention pooling、投影与
L2 归一化——与方案 A 逐行一致。

用法::

    from baseline_multiscale.model import ConvTimeAttentionEncoder

    model = ConvTimeAttentionEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    eeg = model(batch)          # batch: (B, 122, 2501) -> (B, 768)，已 L2 归一化
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# 默认三支核宽（采样点，500 Hz 下为 18 / 50 / 102 ms）
DEFAULT_KERNELS = (9, 25, 51)


def _split_channels(k: int, n_branches: int) -> list[int]:
    """把 k 个特征图尽量平均分给各分支，余数从前往后依次 +1。"""
    base, remainder = divmod(k, n_branches)
    if base == 0:
        raise ValueError(
            f"k={k} 不足以分给 {n_branches} 个分支，每个分支至少需要 1 个通道"
        )
    return [base + (1 if i < remainder else 0) for i in range(n_branches)]


class MultiScaleTemporalConv(nn.Module):
    """并行多支时间卷积，same padding 保证各支输出时间长度一致。

    输入 ``(N, 1, C, T)`` -> 输出 ``(N, k, C, T)``，通道维由各分支拼接而成。

    边界用 ``padding_mode="replicate"``：向外复制最近的采样值，而不是补零，
    避免 same padding 在句子首尾引入阶跃伪瞬变。

    Parameters
    ----------
    k:
        输出特征图总数，按 ``kernels`` 个数平均分配。
    kernels:
        各分支的核宽（采样点），必须为正奇数且不重复。
    """

    def __init__(self, k: int, kernels=DEFAULT_KERNELS):
        super().__init__()

        kernels = tuple(int(m) for m in kernels)
        if not kernels:
            raise ValueError("kernels 不能为空")
        if any(m <= 0 or m % 2 == 0 for m in kernels):
            raise ValueError(f"kernels 必须全为正奇数，收到 {kernels}")
        if len(set(kernels)) != len(kernels):
            raise ValueError(f"kernels 不能重复，收到 {kernels}")

        branch_channels = _split_channels(k, len(kernels))
        self.kernels = kernels
        self.branch_channels = tuple(branch_channels)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                1,
                out_ch,
                kernel_size=(1, m),
                padding=(0, m // 2),
                padding_mode="replicate",
            )
            for out_ch, m in zip(branch_channels, kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 1, C, T)`` -> ``(N, k, C, T)``。"""
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class TSConvFrontend(nn.Module):
    """把整句 ``(B, 1, C, T)`` 压成时间序列 ``(B, k, T2)``。

    与方案 A 的 ``TSConvFrontend`` 结构相同，只把内部的时间卷积换成了
    ``MultiScaleTemporalConv``，并默认使用更短的池化窗（见模块文档）。

    Parameters
    ----------
    n_chans:
        EEG 通道数，本数据集为 122（仅 EEG，已剔除 VEO / HEO / Trigger）。
    k:
        时间卷积的输出通道数，也是后面 attention pooling 的"特征宽度"。
    kernels:
        多尺度核宽（采样点）；传 ``(25,)`` 可退回单支结构。注意本前端用 same
        padding（边界 replicate；方案 A 是 valid padding），所以 T2 会比方案 A
        略长。
    m2 / s:
        时间平均池化的核长与步长，共同决定时间维的下采样倍率 ``s``。
    drop_prob:
        空间卷积之后的 dropout 概率。
    """

    def __init__(
        self,
        n_chans: int = 122,
        k: int = 32,
        kernels=DEFAULT_KERNELS,
        m2: int = 9,
        s: int = 5,
        drop_prob: float = 0.3,
    ):
        super().__init__()

        if n_chans <= 0 or k <= 0:
            raise ValueError("n_chans / k 必须为正")
        if m2 <= 0 or s <= 0:
            raise ValueError("m2 / s 必须为正")
        if not 0 <= drop_prob < 1:
            raise ValueError("drop_prob 必须满足 0 <= drop_prob < 1")

        self.n_chans = n_chans
        self.k = k
        self.m2 = m2
        self.s = s

        self.temporal = MultiScaleTemporalConv(k, kernels)
        self.kernels = self.temporal.kernels
        self.branch_channels = self.temporal.branch_channels
        self.pool = nn.AvgPool2d(kernel_size=(1, m2), stride=(1, s))
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.act = nn.ELU()
        # 空间卷积：一次跨越全部电极，(B, k, C, T2) -> (B, k, 1, T2)
        self.spatial = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=(n_chans, 1))
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=k)
        self.dropout = nn.Dropout(drop_prob)

    def out_time(self, n_samples: int) -> int:
        """给定输入时间长度 ``T``，算出输出时间长度 ``T2``。

        same padding 的时间卷积不改变 T，所以池化前的长度仍是 ``n_samples``。
        """
        if n_samples <= 0:
            raise ValueError("n_samples 必须为正")
        if max(self.kernels) > n_samples:
            raise ValueError(
                f"最大核宽 {max(self.kernels)} 超过输入时间长度 n_samples={n_samples}"
            )
        if n_samples < self.m2:
            raise ValueError(
                f"n_samples={n_samples} 太短：池化窗 m2={self.m2}，至少需要 "
                f"{self.m2} 个采样点"
            )
        return (n_samples - self.m2) // self.s + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, C, T)`` -> ``(B, k, T2)``。"""
        if x.ndim != 4:
            raise ValueError(f"输入必须是 (B, 1, C, T) 四维，收到 {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"期望第 1 维为 1，收到 {x.shape[1]}")
        if x.shape[2] != self.n_chans:
            raise ValueError(f"期望 {self.n_chans} 个 EEG 通道，收到 {x.shape[2]}")

        x = self.temporal(x)            # (B, k, C, T)，same padding
        x = self.pool(x)                # (B, k, C, T2)
        x = self.act(self.norm1(x))
        x = self.spatial(x)             # (B, k, 1, T2)
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        return x.squeeze(2)             # (B, k, T2)


class ConvTimeAttentionEncoder(nn.Module):
    """句子级 EEG 编码器（方案 A' 对照），输出与文本共享空间的定长 embedding。

    Parameters
    ----------
    n_samples:
        输入时间长度 ``T``，必须与数据集给出的形状一致
        （``ChiscoEEGDataset`` 下 reading = 2501、imagine = 1651，500 Hz）。
    n_chans:
        EEG 通道数，本数据集为 122。
    k:
        时间卷积的特征图总数，默认 32（与方案 A 一致，便于对照）。
    kernels:
        多尺度核宽，默认 ``(9, 25, 51)``（18 / 50 / 102 ms）。传 ``(25,)``
        可退回单支结构（same padding，T2 与方案 A 略有差异）。
    m2 / s:
        时间平均池化的核宽与步长，默认 ``(9, 5)``；方案 A 为 ``(25, 5)``。
    d_model / n_heads / dropout:
        注意力池化的配置。默认与方案 A 一致（单层注意力池化，不加 Transformer）。
    embedding_dim:
        最终 embedding 维度；与文本侧对齐，LaBSE 是 768。
    input_scale:
        输入放大系数，默认 1（``ChiscoEEGDataset`` 已返回微伏），含义同方案 A。
    drop_prob:
        空间卷积之后的 dropout 概率。

    Attributes
    ----------
    n_time:
        TSConv 前端输出的时间步数 ``T2``，也是位置编码的长度。
    kernels / branch_channels:
        时间前端实际使用的核宽与各分支通道数，便于日志与消融记录。
    """

    def __init__(
        self,
        n_samples: int,
        n_chans: int = 122,
        k: int = 32,
        kernels=DEFAULT_KERNELS,
        m2: int = 9,
        s: int = 5,
        drop_prob: float = 0.3,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        embedding_dim: int = 768,
        input_scale: float = 1.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 n_heads={n_heads} 整除")
        if input_scale <= 0:
            raise ValueError("input_scale 必须为正")

        self.n_samples = n_samples
        self.n_chans = n_chans
        self.k = k
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.input_scale = input_scale

        self.frontend = TSConvFrontend(
            n_chans=n_chans, k=k, kernels=kernels, m2=m2, s=s, drop_prob=drop_prob
        )
        self.kernels = self.frontend.kernels
        self.branch_channels = self.frontend.branch_channels
        self.n_time = self.frontend.out_time(n_samples)

        self.proj = nn.Linear(k, d_model)

        # 句子的时间顺序就是字序，池化本身对顺序不敏感，位置信息必须显式给出
        self.position = nn.Parameter(torch.zeros(1, self.n_time, d_model))

        # attention pooling：一个可学习 query 对全部时间步做加权汇总
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
        hidden = self.frontend(x)                # (B, k, T2)

        seq = self.proj(hidden.transpose(1, 2))  # (B, T2, d_model)
        seq = seq + self.position

        query = self.pool_query.expand(seq.size(0), -1, -1)
        pooled, _ = self.pool_attn(query, seq, seq)  # (B, 1, d_model)

        return F.normalize(self.out_proj(pooled.squeeze(1)), p=2, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)

    sfreq = 500
    print("尺度划分（500 Hz）:")
    for name, m in zip(("快", "中", "慢"), DEFAULT_KERNELS):
        print(
            f"  {name}支 核宽={m:>2}  "
            f"时长={1000 * m / sfreq:>6.1f} ms  一阶零点≈{sfreq / m:>5.1f} Hz"
        )

    for n_samples in (2501, 1651):
        model = ConvTimeAttentionEncoder(
            n_samples=n_samples, n_chans=122, embedding_dim=768
        )
        total = sum(p.numel() for p in model.parameters())
        print(
            f"\nT={n_samples} -> T2={model.n_time} | 核宽={model.kernels} "
            f"各分支通道={model.branch_channels} | 参数量={total:,}"
        )

    model = ConvTimeAttentionEncoder(n_samples=2501, n_chans=122, embedding_dim=768)
    batch = torch.randn(4, 122, 2501) * 10.0  # 微伏量级，与 ChiscoEEGDataset 的输出一致
    model.eval()
    with torch.no_grad():
        eeg = model(batch)

    print(f"\n输入={tuple(batch.shape)} -> 输出={tuple(eeg.shape)}")
    print("L2 范数:", eeg.norm(dim=-1).tolist())

    # imagine 的 T 不同，必须重建模型
    imagine = ConvTimeAttentionEncoder(n_samples=1651, n_chans=122, embedding_dim=768)
    with torch.no_grad():
        eeg_imagine = imagine(torch.randn(2, 122, 1651) * 10.0)
    print(f"imagine T2={imagine.n_time} -> 输出={tuple(eeg_imagine.shape)}")
