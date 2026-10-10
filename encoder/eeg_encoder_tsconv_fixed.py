"""
固定长度 EEG 的 TSConv Encoder。

适用于：
    - Chisco-1.0
    - 所有 trial 的 EEG 时间长度 T 完全一致
    - 不需要 padding
    - 不需要 padding mask
    - 不需要 AdaptiveAvgPool
    - 不需要 Attention Resampler

整体结构：

    EEG
    (B, C, T)
        ↓
    TSConv Backbone
        ↓
    (B, k, 1, T')
        ↓
    Flatten
        ↓
    (B, k × T')
        ↓
    MLP Projector
        ↓
    projection_hidden_dim
        ↓
    embedding_dim
        ↓
    L2 Normalize
        ↓
    (B, embedding_dim)
"""

import torch
from torch import nn
import torch.nn.functional as F


class TSConvFixedEEGEncoder(nn.Module):
    """
    NICE-inspired TSConv EEG Encoder for fixed-length EEG.

    Parameters
    ----------
    n_chans:
        EEG 通道数。

    n_times:
        每个 EEG trial 的固定采样点数。

        所有输入必须具有相同的 T：

            (B, C, n_times)

    k:
        TSConv 时间卷积输出通道数。

    m1:
        Temporal Convolution 的时间卷积核长度。

    m2:
        Temporal Average Pooling 的时间窗口长度。

    s:
        Temporal Average Pooling 的 stride。

    projection_hidden_dim:
        EEG projection MLP 的隐藏层维度。

    embedding_dim:
        最终 EEG embedding 的维度。

        如果文本 embedding 是 1024 维，
        这里建议设置为 1024。

    drop_prob:
        TSConv Dropout 概率。


    Input
    -----
    eeg:
        shape:

            (B, C, T)

        其中：

            C = n_chans
            T = n_times


    Output
    ------
    embeddings:

        shape:

            (B, embedding_dim)

        最后已经进行了 L2 normalization。
    """

    def __init__(
        self,
        n_chans: int = 122,
        n_times: int = 1651,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
        drop_prob: float = 0.5,
    ):
        super().__init__()

        # ====================================================
        # 1. 参数检查
        # ====================================================

        if not isinstance(n_chans, int) or n_chans <= 0:
            raise ValueError(
                "n_chans must be a positive integer"
            )

        if not isinstance(n_times, int) or n_times <= 0:
            raise ValueError(
                "n_times must be a positive integer"
            )

        if not isinstance(k, int) or k <= 0:
            raise ValueError(
                "k must be a positive integer"
            )

        if not isinstance(m1, int) or m1 <= 0:
            raise ValueError(
                "m1 must be a positive integer"
            )

        if not isinstance(m2, int) or m2 <= 0:
            raise ValueError(
                "m2 must be a positive integer"
            )

        if not isinstance(s, int) or s <= 0:
            raise ValueError(
                "s must be a positive integer"
            )

        if (
            not isinstance(projection_hidden_dim, int)
            or projection_hidden_dim <= 0
        ):
            raise ValueError(
                "projection_hidden_dim must be positive"
            )

        if (
            not isinstance(embedding_dim, int)
            or embedding_dim <= 0
        ):
            raise ValueError(
                "embedding_dim must be positive"
            )

        if not 0 <= drop_prob < 1:
            raise ValueError(
                "drop_prob must satisfy 0 <= drop_prob < 1"
            )

        # ====================================================
        # 2. 保存模型参数
        # ====================================================

        self.n_chans = n_chans
        self.n_times = n_times

        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.s = s

        self.embedding_dim = embedding_dim

        # ====================================================
        # 3. 计算 TSConv 输出时间长度
        # ====================================================
        #
        # 输入：
        #
        #     T = n_times
        #
        # ----------------------------------------------------
        # Temporal Conv:
        #
        # kernel = m1
        # stride = 1
        # padding = 0
        #
        # T1:
        #
        #     T1 = T - m1 + 1
        #
        # ----------------------------------------------------
        # Temporal AvgPool:
        #
        # kernel = m2
        # stride = s
        #
        # T2:
        #
        #     T2 =
        #     floor((T1 - m2) / s) + 1
        #
        # ----------------------------------------------------
        # Spatial Conv:
        #
        # 时间长度不发生变化。
        #
        # 所以最终：
        #
        #     T' = T2
        #
        # ====================================================

        t1 = n_times - m1 + 1

        if t1 < m2:
            min_samples = m1 + m2 - 1

            raise ValueError(
                f"n_times={n_times} is too small. "
                f"At least {min_samples} samples are required "
                f"for m1={m1} and m2={m2}."
            )

        t2 = (
            (t1 - m2) // s
            + 1
        )

        self.output_time_length = t2

        # ====================================================
        # 4. Flatten 后的特征维度
        # ====================================================
        #
        # TSConv 输出：
        #
        #     (B, k, 1, T')
        #
        # Flatten:
        #
        #     (B, k × T')
        #
        # ====================================================

        self.flatten_dim = (
            k
            * self.output_time_length
        )

        # ====================================================
        # 5. TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # Input:
            #
            #     (B, 1, C, T)
            #
            # Output:
            #
            #     (B, k, C, T1)
            #
            # T1:
            #
            #     T - m1 + 1
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=k,
                kernel_size=(1, m1),
                stride=(1, 1),
            ),

            # ------------------------------------------------
            # Temporal Average Pooling
            #
            # Input:
            #
            #     (B, k, C, T1)
            #
            # Output:
            #
            #     (B, k, C, T2)
            #
            # T2:
            #
            #     floor((T1 - m2) / s) + 1
            # ------------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, m2),
                stride=(1, s),
            ),

            nn.BatchNorm2d(
                k
            ),

            nn.ELU(),

            # ------------------------------------------------
            # Spatial Convolution
            #
            # kernel_size:
            #
            #     (n_chans, 1)
            #
            # 因此卷积核会一次跨越全部 EEG 电极。
            #
            # Input:
            #
            #     (B, k, C, T2)
            #
            # Output:
            #
            #     (B, k, 1, T2)
            #
            # 时间长度不改变。
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
            ),

            nn.BatchNorm2d(
                k
            ),

            nn.ELU(),

            nn.Dropout(
                p=drop_prob
            ),
        )

        # ====================================================
        # 6. EEG → Text Shared Embedding Space
        # ====================================================
        #
        # TSConv:
        #
        #     (B, k, 1, T')
        #
        # Flatten:
        #
        #     (B, k × T')
        #
        # MLP:
        #
        #     flatten_dim
        #         ↓
        #     projection_hidden_dim
        #         ↓
        #     embedding_dim
        #
        # 默认：
        #
        #     ? → 512 → 1024
        #
        # ====================================================

        self.eeg_projection = nn.Sequential(

            nn.Linear(
                self.flatten_dim,
                projection_hidden_dim,
            ),

            nn.GELU(),

            nn.Linear(
                projection_hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward.

        Parameters
        ----------
        eeg:
            EEG 输入。

            shape:

                (B, C, T)

            必须满足：

                C == n_chans

                T == n_times


        Returns
        -------
        torch.Tensor:

            L2-normalized EEG embedding。

            shape:

                (B, embedding_dim)
        """

        # ====================================================
        # 1. 输入维度检查
        # ====================================================

        if eeg.ndim != 3:
            raise ValueError(
                "EEG input must have shape "
                "(B, C, T), "
                f"but got {tuple(eeg.shape)}"
            )

        # ====================================================
        # 2. EEG 通道数检查
        # ====================================================

        if eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected {self.n_chans} EEG channels, "
                f"but got {eeg.shape[1]}"
            )

        # ====================================================
        # 3. EEG 时间长度检查
        # ====================================================
        #
        # Fixed Encoder 最关键的一点：
        #
        # 所有 trial 必须具有完全相同的 T。
        #
        # ====================================================

        if eeg.shape[-1] != self.n_times:
            raise ValueError(
                f"Expected EEG length T={self.n_times}, "
                f"but got T={eeg.shape[-1]}. "
                "TSConvFixedEEGEncoder requires "
                "fixed-length EEG."
            )

        # ====================================================
        # 4. 增加 Conv2d 输入通道维度
        # ====================================================
        #
        # 原始：
        #
        #     (B, C, T)
        #
        # 变成：
        #
        #     (B, 1, C, T)
        #
        # ====================================================

        eeg = eeg.unsqueeze(
            dim=1
        )

        # ====================================================
        # 5. TSConv Backbone
        # ====================================================
        #
        # Input:
        #
        #     (B, 1, C, T)
        #
        # Output:
        #
        #     (B, k, 1, T')
        #
        # ====================================================

        features = self.tsconv(
            eeg
        )

        # ====================================================
        # 6. Flatten
        # ====================================================
        #
        # (B, k, 1, T')
        #
        # →
        #
        # (B, k × T')
        #
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 7. MLP Projection
        # ====================================================
        #
        # flatten_dim
        #
        # →
        #
        # projection_hidden_dim
        #
        # →
        #
        # embedding_dim
        #
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 8. L2 Normalize
        # ====================================================
        #
        # z_hat:
        #
        #     z / ||z||_2
        #
        # 如果 Text embedding 同样进行 L2 normalization：
        #
        #     eeg_embedding @ text_embedding.T
        #
        # 就对应 cosine similarity。
        #
        # ====================================================

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

        return embeddings


# ============================================================
# Demo
# ============================================================


if __name__ == "__main__":

    # ========================================================
    # 1. 创建模型
    # ========================================================
    #
    # 注意：
    #
    # 这里的 n_chans 和 n_times
    # 应该与你实际处理后的 Chisco-1.0 完全一致。
    #
    # ========================================================

    model = TSConvFixedEEGEncoder(

        # EEG
        n_chans=122,
        n_times=1651,

        # NICE TSConv
        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # EEG → Text Projection
        projection_hidden_dim=512,
        embedding_dim=1024,
    )

    # ========================================================
    # 2. 打印模型结构信息
    # ========================================================

    print(
        "========================================"
    )

    print(
        f"Input EEG shape:"
        f" (B, {model.n_chans}, {model.n_times})"
    )

    print(
        f"TSConv output time length:"
        f" {model.output_time_length}"
    )

    print(
        f"Flatten dimension:"
        f" {model.flatten_dim}"
    )

    print(
        f"Embedding dimension:"
        f" {model.embedding_dim}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 3. 创建模拟 EEG
    # ========================================================

    batch_size = 8

    eeg = torch.randn(
        batch_size,
        model.n_chans,
        model.n_times,
    )

    # ========================================================
    # 4. Forward
    # ========================================================

    model.eval()

    with torch.no_grad():

        embeddings = model(
            eeg
        )

    # ========================================================
    # 5. 查看 Shape
    # ========================================================

    print(
        "\nEEG:"
    )

    print(
        eeg.shape
    )

    print(
        "\nEmbedding:"
    )

    print(
        embeddings.shape
    )

    # ========================================================
    # 6. 检查 L2 Norm
    # ========================================================

    norms = torch.linalg.vector_norm(
        embeddings,
        dim=-1,
    )

    print(
        "\nEmbedding L2 norms:"
    )

    print(
        norms
    )

    # ========================================================
    # 7. 参数量
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\n========================================"
    )

    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        "========================================"
    )