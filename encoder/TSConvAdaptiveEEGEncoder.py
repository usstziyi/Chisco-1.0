import torch
from torch import nn
import torch.nn.functional as F


class TSConvAdaptiveEEGEncoder(nn.Module):
    """
    NICE-inspired TSConv EEG Encoder
    支持不同 batch 使用不同 EEG 时间长度。

    训练方式:
        - 按 EEG 长度建立 Length Buckets
        - 同一个 batch 内所有 EEG 长度相同
        - 不同长度的 batch 随机交替训练
        - 不使用 padding
        - 不使用 padding mask

    输入:
        eeg: (B, C, T)

    其中:
        B = batch size
        C = EEG 通道数
        T = 当前 batch 的 EEG 时间长度

    注意:
        T 可以在不同 batch 之间变化，
        但同一个 batch 内必须一致，因为 PyTorch Tensor
        本身要求 batch 内 shape 一致。

    整体结构:

        EEG
        (B, C, T)
            ↓
        TSConv Backbone
            ↓
        (B, k, 1, T')
            ↓
        AdaptiveAvgPool2d
        T' → adaptive_bins
            ↓
        (B, k, 1, adaptive_bins)
            ↓
        Flatten
            ↓
        (B, k × adaptive_bins)
            ↓
        MLP Projector
        k × adaptive_bins
            ↓
        projection_hidden_dim
            ↓
        embedding_dim
            ↓
        L2 Normalize
            ↓
        (B, embedding_dim)
    """

    def __init__(
        self,
        n_chans: int = 125,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        adaptive_bins: int = 64,
        projection_hidden_dim: int = 512,
        embedding_dim: int = 1024,
        drop_prob: float = 0.5,
    ):
        super().__init__()

        # ====================================================
        # 1. 参数检查
        # ====================================================

        if n_chans <= 0:
            raise ValueError("n_chans must be positive")

        if k <= 0:
            raise ValueError("k must be positive")

        if m1 <= 0:
            raise ValueError("m1 must be positive")

        if m2 <= 0:
            raise ValueError("m2 must be positive")

        if s <= 0:
            raise ValueError("s must be positive")

        if adaptive_bins <= 0:
            raise ValueError("adaptive_bins must be positive")

        if projection_hidden_dim <= 0:
            raise ValueError(
                "projection_hidden_dim must be positive"
            )

        if embedding_dim <= 0:
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

        # TSConv 输出特征通道数
        self.k = k

        self.m1 = m1
        self.m2 = m2
        self.s = s

        # Adaptive Pooling 后固定保留的时间位置数
        self.adaptive_bins = adaptive_bins

        # 最终 EEG/Text 共享 embedding 维度
        self.embedding_dim = embedding_dim

        # Adaptive Pooling 后 Flatten 的维度固定为:
        #
        # k × adaptive_bins
        #
        # 默认:
        #
        # 40 × 64 = 2560
        self.flatten_dim = (
            k * adaptive_bins
        )

        # ====================================================
        # 3. TSConv Backbone
        # ====================================================

        self.tsconv = nn.Sequential(

            # ------------------------------------------------
            # Temporal Convolution
            #
            # 输入:
            # (B, 1, C, T)
            #
            # 输出:
            # (B, k, C, T1)
            #
            # T1 = T - m1 + 1
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
            # 输入:
            # (B, k, C, T1)
            #
            # 输出:
            # (B, k, C, T2)
            #
            # T2 =
            # floor((T1 - m2) / s) + 1
            #
            # T2 可以随着不同 batch 的 EEG 长度变化。
            # ------------------------------------------------

            nn.AvgPool2d(
                kernel_size=(1, m2),
                stride=(1, s),
            ),

            nn.BatchNorm2d(k),

            nn.ELU(),

            # ------------------------------------------------
            # Spatial Convolution
            #
            # 卷积核一次跨越所有 EEG 电极。
            #
            # 输入:
            # (B, k, C, T2)
            #
            # 输出:
            # (B, k, 1, T2)
            #
            # 时间长度 T2 不发生变化。
            # ------------------------------------------------

            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=(n_chans, 1),
                stride=(1, 1),
            ),

            nn.BatchNorm2d(k),

            nn.ELU(),

            nn.Dropout(
                p=drop_prob
            ),
        )

        # ====================================================
        # 4. Adaptive Temporal Pooling
        # ====================================================
        #
        # 不管 TSConv 输出时间长度 T2 是多少:
        #
        # (B, k, 1, T2)
        #
        # 都统一变成:
        #
        # (B, k, 1, adaptive_bins)
        #
        # 默认:
        #
        # adaptive_bins = 64
        #
        # 因此不同长度 EEG 最终都能进入同一个 MLP。
        # ====================================================

        self.temporal_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, adaptive_bins)
        )

        # ====================================================
        # 5. EEG → Text Shared Space Projector
        # ====================================================
        #
        # Adaptive Pool:
        #
        # (B, k, 1, adaptive_bins)
        #
        # Flatten:
        #
        # (B, k × adaptive_bins)
        #
        # 默认:
        #
        # 40 × 64 = 2560
        #
        # MLP:
        #
        # 2560 → 512 → 1024
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
            EEG 输入:

            (B, C, T)

            T 可以在不同 batch 之间变化。

        Returns
        -------
        torch.Tensor:
            L2-normalized EEG embedding:

            (B, embedding_dim)
        """

        # ====================================================
        # 1. 输入检查
        # ====================================================

        if eeg.ndim != 3:
            raise ValueError(
                "EEG input must have shape "
                "(B, C, T), "
                f"but got {tuple(eeg.shape)}"
            )

        if eeg.shape[1] != self.n_chans:
            raise ValueError(
                f"Expected {self.n_chans} EEG channels, "
                f"but got {eeg.shape[1]}"
            )

        # ----------------------------------------------------
        # TSConv 能正常工作的最小时间长度
        #
        # Temporal Conv 后必须至少剩下 m2 个点，
        # AvgPool 才能运行:
        #
        # T - m1 + 1 >= m2
        #
        # 因此:
        #
        # T >= m1 + m2 - 1
        #
        # NICE 默认:
        #
        # 25 + 51 - 1 = 75
        # ----------------------------------------------------

        min_samples = (
            self.m1
            + self.m2
            - 1
        )

        if eeg.shape[-1] < min_samples:
            raise ValueError(
                f"EEG sequence is too short: "
                f"got {eeg.shape[-1]} samples, "
                f"but at least {min_samples} are required."
            )

        # ====================================================
        # 2. 增加 Conv2d 输入通道维
        # ====================================================
        #
        # (B, C, T)
        #
        # →
        #
        # (B, 1, C, T)
        # ====================================================

        eeg = eeg.unsqueeze(1)

        # ====================================================
        # 3. TSConv Backbone
        # ====================================================
        #
        # (B, 1, C, T)
        #
        # →
        #
        # (B, k, 1, T')
        #
        # T' 会随着当前 batch 的 EEG 长度变化。
        # ====================================================

        features = self.tsconv(
            eeg
        )

        # ====================================================
        # 4. Adaptive Temporal Pooling
        # ====================================================
        #
        # 例如:
        #
        # 短 EEG:
        # (B, 40, 1, 386)
        #
        # 长 EEG:
        # (B, 40, 1, 706)
        #
        # 都变成:
        #
        # (B, 40, 1, 64)
        # ====================================================

        features = self.temporal_pool(
            features
        )

        # ====================================================
        # 5. Flatten
        # ====================================================
        #
        # (B, k, 1, adaptive_bins)
        #
        # →
        #
        # (B, k × adaptive_bins)
        #
        # 默认:
        #
        # (B, 40, 1, 64)
        #
        # →
        #
        # (B, 2560)
        # ====================================================

        features = features.flatten(
            start_dim=1
        )

        # ====================================================
        # 6. MLP Projection Head
        # ====================================================
        #
        # 默认:
        #
        # (B, 2560)
        #
        # →
        #
        # (B, 512)
        #
        # →
        #
        # (B, 1024)
        # ====================================================

        embeddings = self.eeg_projection(
            features
        )

        # ====================================================
        # 7. L2 Normalize
        # ====================================================
        #
        # z_hat = z / ||z||_2
        #
        # 如果 Text embedding 也做了 L2 normalization，
        # 则两边点积就是 cosine similarity。
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
    # 创建一个共享 EEGEncoder
    #
    # 注意:
    # 模型只创建一次。
    #
    # 不同 EEG 长度的 batch 都进入同一个 model。
    # ========================================================

    model = TSConvAdaptiveEEGEncoder(
        n_chans=125,

        # NICE TSConv
        k=40,
        m1=25,
        m2=51,
        s=5,
        drop_prob=0.5,

        # Adaptive Pool
        adaptive_bins=64,
        
        # MLP
        projection_hidden_dim=512,

        # Text embedding dimension
        embedding_dim=1024,
    )

    # ========================================================
    # 模拟三个不同长度 bucket
    # ========================================================

    batch_size = 8

    eeg_short = torch.randn(
        batch_size,
        125,
        2000,
    )

    eeg_medium = torch.randn(
        batch_size,
        125,
        2800,
    )

    eeg_long = torch.randn(
        batch_size,
        125,
        3600,
    )

    # ========================================================
    # 同一个模型处理不同长度 EEG
    # ========================================================

    embedding_short = model(
        eeg_short
    )

    embedding_medium = model(
        eeg_medium
    )

    embedding_long = model(
        eeg_long
    )

    # ========================================================
    # 输出 shape
    # ========================================================

    print("Short EEG:")
    print(eeg_short.shape)
    print(embedding_short.shape)

    print("\nMedium EEG:")
    print(eeg_medium.shape)
    print(embedding_medium.shape)

    print("\nLong EEG:")
    print(eeg_long.shape)
    print(embedding_long.shape)

    # ========================================================
    # 检查 L2 norm
    # ========================================================

    print("\nShort embedding norms:")
    print(
        torch.linalg.vector_norm(
            embedding_short,
            dim=-1,
        )
    )

    print("\nMedium embedding norms:")
    print(
        torch.linalg.vector_norm(
            embedding_medium,
            dim=-1,
        )
    )

    print("\nLong embedding norms:")
    print(
        torch.linalg.vector_norm(
            embedding_long,
            dim=-1,
        )
    )