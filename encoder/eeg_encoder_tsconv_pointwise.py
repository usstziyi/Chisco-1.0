"""
TSConv + Pointwise Conv EEG Encoder.

适用于固定长度 EEG，例如 Chisco-1.0：

    单条 EEG:
        (125, 1651)

整体结构：

    EEG
    (B, C, T)
        ↓
    TSConv Backbone
        ↓
    (B, k, 1, T')
        ↓
    1×1 Pointwise Conv
    k → pointwise_channels
        ↓
    (B, pointwise_channels, 1, T')
        ↓
    Flatten
        ↓
    (B, pointwise_channels × T')
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

特点：

    - EEG 为固定长度
    - 不使用 padding
    - 不使用 padding mask
    - 不使用 AdaptiveAvgPool
    - 不使用 Attention Resampler
    - 使用 1×1 Conv 压缩 TSConv feature channels
"""

import torch
from torch import nn
import torch.nn.functional as F


class TSConvPointwiseEEGEncoder(nn.Module):
    """
    NICE-inspired TSConv EEG Encoder
    + 1×1 Pointwise Convolution。

    适用于固定长度 EEG。

    Parameters
    ----------
    n_chans:
        EEG 通道数。

    n_times:
        每条 EEG trial 的固定采样点数。

    k:
        TSConv Temporal Convolution 输出通道数。

    m1:
        Temporal Convolution 的时间卷积核长度。

    m2:
        Temporal Average Pooling 的时间窗口长度。

    s:
        Temporal Average Pooling 的 stride。

    pointwise_channels:
        1×1 Conv 输出通道数。

        例如：

            40 → 8

        可以显著降低 Flatten 后的特征维度。

    projection_hidden_dim:
        EEG Projector 的隐藏层维度。

    embedding_dim:
        最终 EEG embedding 维度。

        如果文本 embedding 为 1024 维，
        这里设置为 1024。

    drop_prob:
        TSConv 中的 Dropout 概率。


    Input
    -----
    eeg:

        (B, C, T)

        默认：

            (B, 125, 1651)


    Output
    ------
    embeddings:

        (B, embedding_dim)

        默认：

            (B, 1024)

        输出已经进行 L2 normalization。
    """

    def __init__(
        self,
        n_chans: int = 125,
        n_times: int = 1651,

        # TSConv
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        drop_prob: float = 0.5,

        # Pointwise Conv
        pointwise_channels: int = 8,

        # Projection
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
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
            not isinstance(pointwise_channels, int)
            or pointwise_channels <= 0
        ):
            raise ValueError(
                "pointwise_channels must be a positive integer"
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
        # 2. 保存参数
        # ====================================================

        self.n_chans = n_chans
        self.n_times = n_times

        self.k = k

        self.m1 = m1
        self.m2 = m2
        self.s = s

        self.pointwise_channels = pointwise_channels

        self.embedding_dim = embedding_dim

        # ====================================================
        # 3. 计算 TSConv 输出时间长度
        # ====================================================
        #
        # 输入：
        #
        #     T = n_times
        #
        # Temporal Conv:
        #
        #     kernel = m1
        #     stride = 1
        #     padding = 0
        #
        # 所以：
        #
        #     T1 = T - m1 + 1
        #
        #
        # Temporal AvgPool:
        #
        #     kernel = m2
        #     stride = s
        #
        # 所以：
        #
        #     T2 =
        #     floor((T1 - m2) / s) + 1
        #
        #
        # Spatial Conv:
        #
        #     不改变时间长度
        #
        # 最终：
        #
        #     T' = T2
        #
        # ====================================================

        t1 = n_times - m1 + 1

        if t1 < m2:

            min_samples = (
                m1
                + m2
                - 1
            )

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
        # 4. Pointwise Conv 后 Flatten 的维度
        # ====================================================
        #
        # TSConv:
        #
        #     (B, k, 1, T')
        #
        # Pointwise Conv:
        #
        #     (B, pointwise_channels, 1, T')
        #
        # Flatten:
        #
        #     (B, pointwise_channels × T')
        #
        #
        # 默认：
        #
        #     T' = 316
        #
        #     pointwise_channels = 8
        #
        # 所以：
        #
        #     8 × 316 = 2528
        #
        # ====================================================

        self.flatten_dim = (
            pointwise_channels
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
            # 默认：
            #
            #     (B, 1, 125, 1651)
            #
            #
            # Output:
            #
            #     (B, k, C, T1)
            #
            # 默认：
            #
            #     (B, 40, 125, 1627)
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
            # 默认：
            #
            #     (B, 40, 125, 316)
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
            # 卷积核在空间方向一次跨越全部 EEG 电极：
            #
            #     kernel_size=(n_chans, 1)
            #
            #
            # Input:
            #
            #     (B, k, C, T2)
            #
            # Output:
            #
            #     (B, k, 1, T2)
            #
            #
            # 默认：
            #
            #     (B, 40, 125, 316)
            #
            # →
            #
            #     (B, 40, 1, 316)
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
        # 6. 1×1 Pointwise Convolution
        # ====================================================
        #
        # 这里不是改变时间长度，
        #
        # 而是在每一个时间位置上，
        # 对 TSConv 的 k 个 feature maps
        # 做可学习的线性组合。
        #
        #
        # Input:
        #
        #     (B, k, 1, T')
        #
        # 默认：
        #
        #     (B, 40, 1, 316)
        #
        #
        # Output:
        #
        #     (B, pointwise_channels, 1, T')
        #
        # 默认：
        #
        #     (B, 8, 1, 316)
        #
        # ====================================================

        self.pointwise_projection = nn.Sequential(

            nn.Conv2d(
                in_channels=k,
                out_channels=pointwise_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),

            nn.BatchNorm2d(
                pointwise_channels
            ),

            nn.ELU(),
        )

        # ====================================================
        # 7. EEG → Text Shared Embedding Space
        # ====================================================
        #
        # Pointwise Conv:
        #
        #     (B, 8, 1, 316)
        #
        # Flatten:
        #
        #     (B, 2528)
        #
        #
        # MLP:
        #
        #     2528
        #       ↓
        #     512
        #       ↓
        #     1024
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
        Parameters
        ----------
        eeg:

            EEG input

            shape:

                (B, C, T)

            默认：

                (B, 125, 1651)


        Returns
        -------
        torch.Tensor:

            L2-normalized EEG embedding

            shape:

                (B, embedding_dim)

            默认：

                (B, 1024)
        """

        # ====================================================
        # 1. 输入 Shape 检查
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
        # 本模型是 Fixed-Length Encoder。
        #
        # 所有 trial 的 T 必须完全一样。
        #
        # ====================================================

        if eeg.shape[-1] != self.n_times:
            raise ValueError(
                f"Expected EEG length T={self.n_times}, "
                f"but got T={eeg.shape[-1]}. "
                "TSConvPointwiseEEGEncoder requires "
                "fixed-length EEG."
            )

        # ====================================================
        # 4. 增加 Conv2d 输入通道维度
        # ====================================================
        #
        # (B, C, T)
        #
        # →
        #
        # (B, 1, C, T)
        #
        # 默认：
        #
        # (B, 125, 1651)
        #
        # →
        #
        # (B, 1, 125, 1651)
        #
        # ====================================================

        eeg = eeg.unsqueeze(
            dim=1
        )

        # ====================================================
        # 5. TSConv
        # ====================================================
        #
        # (B, 1, 125, 1651)
        #
        # →
        #
        # (B, 40, 1, 316)
        #
        # ====================================================

        features = self.tsconv(
            eeg
        )

        # ====================================================
        # 6. 1×1 Pointwise Conv
        # ====================================================
        #
        # (B, 40, 1, 316)
        #
        # →
        #
        # (B, 8, 1, 316)
        #
        # ====================================================

        features = self.pointwise_projection(
            features
        )

        # ====================================================
        # 7. Flatten
        # ====================================================
        #
        # (B, 8, 1, 316)
        #
        # →
        #
        # (B, 2528)
        #
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 8. MLP Projection
        # ====================================================
        #
        # (B, 2528)
        #
        # →
        #
        # (B, 512)
        #
        # →
        #
        # (B, 1024)
        #
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 9. L2 Normalize
        # ====================================================
        #
        # z_hat:
        #
        #     z / ||z||_2
        #
        #
        # 如果 Text embedding 同样进行 L2 normalization，
        #
        # 那么：
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

    model = TSConvPointwiseEEGEncoder(

        # ----------------------------------------------------
        # Chisco-1.0 EEG
        # ----------------------------------------------------

        n_chans=125,
        n_times=1651,

        # ----------------------------------------------------
        # TSConv
        # ----------------------------------------------------

        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # ----------------------------------------------------
        # 1×1 Pointwise Conv
        # ----------------------------------------------------

        pointwise_channels=8,

        # ----------------------------------------------------
        # EEG → Text Projection
        # ----------------------------------------------------

        projection_hidden_dim=512,
        embedding_dim=1024,
    )

    # ========================================================
    # 2. 查看模型关键参数
    # ========================================================

    print(
        "========================================"
    )

    print(
        "TSConvPointwiseEEGEncoder"
    )

    print(
        "========================================"
    )

    print(
        f"Input EEG shape: "
        f"(B, {model.n_chans}, {model.n_times})"
    )

    print(
        f"TSConv channels: "
        f"{model.k}"
    )

    print(
        f"TSConv output time length: "
        f"{model.output_time_length}"
    )

    print(
        f"Pointwise channels: "
        f"{model.pointwise_channels}"
    )

    print(
        f"Flatten dimension: "
        f"{model.flatten_dim}"
    )

    print(
        f"Embedding dimension: "
        f"{model.embedding_dim}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 3. 模拟一个 Batch
    # ========================================================

    batch_size = 8

    eeg = torch.randn(
        batch_size,
        125,
        1651,
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
    # 5. 查看输出 Shape
    # ========================================================

    print(
        "\nInput EEG:"
    )

    print(
        eeg.shape
    )

    print(
        "\nOutput embedding:"
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