import os
from pathlib import Path

# Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Windows 无符号链接权限提示
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 加载 LaBSE
# ============================================================

model = SentenceTransformer(
    "sentence-transformers/LaBSE"
)


# ============================================================
# 2. 输入句子
# ============================================================

sentences = [
    "我喜欢人工智能",
    "I love artificial intelligence",
    "今天天气很好"
]


# ============================================================
# 3. 生成句子 embedding
# ============================================================

embeddings = model.encode(
    sentences,
    normalize_embeddings=False
)

# 转为 FP32
embeddings = embeddings.astype(np.float32)

# 手动进行 L2 normalization
embeddings /= np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

print("Embedding shape:")
print(embeddings.shape)

print("\nEmbedding dtype:")
print(embeddings.dtype)

print("\nEmbedding norms:")
print(np.linalg.norm(embeddings, axis=1))


# ============================================================
# 4. 计算余弦相似度矩阵
# ============================================================

# 因为 embedding 已经 L2 normalize：
#
# cosine_similarity(a, b) = a @ b
#
similarity = embeddings @ embeddings.T

print("\nSimilarity matrix:")
print(similarity)


# ============================================================
# 5. 绘制相似度热力图
# ============================================================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ax = sns.heatmap(
    similarity,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    xticklabels=sentences,
    yticklabels=sentences,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={
        "shrink": 0.8,
        "label": "Cosine Similarity"
    },
    annot_kws={
        "size": 11
    }
)

# 深色格子使用白字，浅色格子使用黑字
for text in ax.texts:
    value = float(text.get_text())
    text.set_color(
        "white" if value > 0.6 else "black"
    )

plt.title("LaBSE 句子余弦相似度矩阵")

plt.tight_layout()


# ============================================================
# 6. 保存
# ============================================================

out_dir = (
    Path(__file__).resolve().parents[1]
    / "outputs"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    out_dir
    / "labse_similarity_matrix.png"
)

plt.savefig(
    output_path,
    dpi=300
)

print(f"\nSaved to: {output_path}")

plt.show()
