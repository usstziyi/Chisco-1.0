"""
LaBSE 文本编码器。

适用于：

    - Chisco-1.0 的句子文本
    - 需要把文本映射到与 EEG encoder 共享的 embedding 空间
    - 训练时冻结，只做 forward

整体结构：

    句子
    (list[str], 长度 B)
        ↓
    LaBSE (SentenceTransformer)
        ↓
    (B, D)
        ↓
    L2 Normalize
        ↓
    (B, D)

    其中 D 由模型决定：

        LaBSE → 768

特点：

    - 输出维度固定，由模型决定
    - 默认冻结全部参数，不参与梯度更新
    - 输出已经 L2 normalization
    - encode_unique 自带文本缓存，同一条文本只编码一次
    - 如果 EEG encoder 的输出同样经过 L2 normalization，
      那么：

          eeg_embedding @ text_embedding.T

      就对应 cosine similarity
    - sentence_transformers 为延迟导入，
      import 本模块不会加载模型、不产生副作用
"""

import os

import torch
from torch import nn
import torch.nn.functional as F


class LaBSETextEncoder(nn.Module):
    """
    LaBSE 句子 → 文本 embedding 编码器。

    Parameters
    ----------
    model_name:
        SentenceTransformer 模型名或本地路径。

        默认：

            "sentence-transformers/LaBSE"

    Input
    -----
    texts:

        list[str]，长度为 B。

        也允许传入单个 str，此时按 B = 1 处理。

    Output
    ------
    embeddings:

        (B, embedding_dim)

        LaBSE 默认：

            (B, 768)

        输出已经进行 L2 normalization。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
    ):
        super().__init__()

        # ====================================================
        # 1. 参数检查
        # ====================================================

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string"
            )

        # ====================================================
        # 2. 保存参数
        # ====================================================

        self.model_name = model_name

        # 文本 → embedding 缓存，供 encode_unique 使用
        self.cache: dict[str, torch.Tensor] = {}

        # ====================================================
        # 3. Hugging Face 镜像
        # ====================================================
        #
        # 必须在导入 sentence_transformers 之前设置，
        # 否则 huggingface_hub 已经读走了默认 endpoint。
        #
        # 使用 setdefault，不覆盖用户已有的配置。
        #
        # ====================================================

        os.environ.setdefault(
            "HF_ENDPOINT",
            "https://hf-mirror.com",
        )

        # Windows 无符号链接权限提示
        os.environ.setdefault(
            "HF_HUB_DISABLE_SYMLINKS_WARNING",
            "1",
        )

        # ====================================================
        # 4. 延迟导入并加载 LaBSE
        # ====================================================

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name
        )

        # ====================================================
        # 5. 读取 embedding 维度
        # ====================================================

        actual_dim = self.model.get_embedding_dimension()

        if actual_dim is None:
            raise ValueError(
                f"Cannot determine embedding dimension of "
                f"{model_name!r}"
            )

        self._embedding_dim = int(actual_dim)

        # ====================================================
        # 6. 冻结参数
        # ====================================================
        #
        # 文本侧在训练中不更新：
        #
        #     eval() 固定 Dropout / BatchNorm 行为
        #     requires_grad = False 不建梯度图
        #
        # ====================================================

        self.model.eval() # Dropout/BatchNorm 开关

        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        """模型实际的输出维度。"""
        return self._embedding_dim

    def forward(
        self,
        texts: str | list[str],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        texts:

            单个 str 或 list[str]。

        Returns
        -------
        torch.Tensor:

            L2-normalized 文本 embedding。

            shape:

                (B, embedding_dim)
        """

        # ====================================================
        # 1. 统一输入形式：str → list[str]
        # ====================================================
        #
        # 统一成 list[str]：
        #
        #     "hello"             → ["hello"]
        #     ["a", "b"]          → ["a", "b"]
        #
        # ====================================================

        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            raise ValueError(
                "texts must not be empty"
            )

        if not all(isinstance(t, str) for t in texts):
            raise TypeError(
                "every element of texts must be a str"
            )

        # ====================================================
        # 2. 文本编码
        # ====================================================
        #
        # 冻结编码器，不需要 autograd。
        #
        # normalize_embeddings=False：
        #
        #     它只是 encode() 自己的一个开关，管的是出口那一步：
        #
        #         if normalize_embeddings:
        #             embeddings = F.normalize(embeddings)
        #
        #     它关不掉 LaBSE Pipeline 内部的 Normalize 模块。
        #
        #     LaBSE 的模块链：
        #
        #         Transformer → Pooling → Dense → Normalize
        #
        #     末尾的 Normalize 是模型的一个 nn.Module，
        #     在 forward 遍历模块时无条件执行。
        #
        #     所以：
        #
        #         这里拿到的 embedding 本来就是单位向量。
        #
        #     下面第 5 步还会再归一化一次：
        #
        #         对单位向量再归一化是恒等变换，结果不变。
        #
        #     之所以仍然保留第 5 步：
        #
        #         - 输出契约由自己持有，
        #           换成没有 Normalize 模块的模型也成立
        #         - 与 EEG encoder 的处理方式保持一致
        #
        # ====================================================

        with torch.no_grad():

            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        # ====================================================
        # 3. 转为 FP32
        # ====================================================

        embeddings = embeddings.float()

        # ====================================================
        # 4. 输出维度检查
        # ====================================================

        if embeddings.ndim != 2:
            raise ValueError(
                "text embeddings must be 2-dimensional, "
                f"but got shape {tuple(embeddings.shape)}"
            )

        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dim "
                f"{self.embedding_dim}, "
                f"but got {embeddings.shape[-1]}"
            )

        # ====================================================
        # 5. L2 Normalize
        # ====================================================
        #
        # z_hat:
        #
        #     z / ||z||_2
        #
        # 如果 EEG embedding 同样进行 L2 normalization：
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

    @torch.no_grad() # 关闭梯度记录
    def encode_unique(
        self,
        texts: str | list[str],
    ) -> torch.Tensor:
        """
        带缓存的文本编码。

        已经编码过的文本直接命中缓存，只对没见过的文本调用一次
        LaBSE。多 epoch 训练下每条文本只编码一次。

        Parameters
        ----------
        texts:

            单个 str 或 list[str]。

        Returns
        -------
        torch.Tensor:

            L2-normalized 文本 embedding。

            shape:

                (B, embedding_dim)

            顺序与输入 texts 完全一致。
        """

        # ====================================================
        # 1. 统一输入形式：str → list[str]
        # ====================================================

        if isinstance(texts, str):
            texts = [texts]

        # ====================================================
        # 2. 找出没见过的文本
        # ====================================================
        #
        # dict.fromkeys 去重，保持首次出现的顺序：
        #
        #     ["b", "a", "b"]  →  ["b", "a"]
        #
        # ====================================================

        unseen = [
            text
            for text in dict.fromkeys(texts)
            if text not in self.cache
        ]

        # ====================================================
        # 3. 编码没见过的文本并写入缓存
        # ====================================================
        #
        # clone() 让每条缓存独立，避免存的是
        # 某个 batch 大张量的 view，从而拖住整块内存。
        #
        # ====================================================

        if unseen:

            embeddings = self.forward(unseen)

            for text, vector in zip(unseen, embeddings):
                self.cache[text] = vector.clone()

        # ====================================================
        # 4. 按输入顺序取回
        # ====================================================

        return torch.stack([self.cache[text] for text in texts])


# ============================================================
# Demo
# ============================================================


if __name__ == "__main__":

    # ========================================================
    # 1. 创建编码器
    # ========================================================

    encoder = LaBSETextEncoder(
        model_name="sentence-transformers/LaBSE",
    )

    # ========================================================
    # 2. 打印模型信息
    # ========================================================

    print(
        "========================================"
    )

    print(
        "LaBSETextEncoder"
    )

    print(
        "========================================"
    )

    print(
        f"Model name: "
        f"{encoder.model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{encoder.embedding_dim}"
    )

    print(
        f"Frozen: "
        f"{not any(p.requires_grad for p in encoder.parameters())}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # 3. 输入句子
    # ========================================================

    sentences = [
        "我喜欢人工智能",
        "I love artificial intelligence",
        "今天天气很好",
    ]

    # ========================================================
    # 4. Forward
    # ========================================================

    embeddings = encoder(
        sentences
    )

    # ========================================================
    # 5. 查看 Shape / dtype
    # ========================================================

    print(
        "\nSentences:"
    )

    print(
        sentences
    )

    print(
        "\nEmbedding shape:"
    )

    print(
        embeddings.shape
    )

    print(
        "\nEmbedding dtype:"
    )

    print(
        embeddings.dtype
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
    # 7. 余弦相似度矩阵
    # ========================================================
    #
    # embedding 已经 L2 normalize：
    #
    #     cosine_similarity(a, b) = a @ b
    #
    # ========================================================

    similarity = embeddings @ embeddings.T

    print(
        "\nSimilarity matrix:"
    )

    print(
        similarity
    )

    # ========================================================
    # 8. 参数量
    # ========================================================

    total_params = sum(
        p.numel()
        for p in encoder.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in encoder.parameters()
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
