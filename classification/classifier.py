"""在 baseline 的句子级 EEG 编码器上接一个线性分类头。

    (B, C, T)
        ↓  ConvTimeAttentionEncoder（baseline/model.py，输出 L2 归一化 embedding）
    (B, embedding_dim)
        ↓  Dropout
        ↓  Linear(embedding_dim -> n_classes)
        ↓  x logit_scale
    (B, n_classes)   logits

``ConvTimeAttentionEncoder`` 的输出是单位向量，直接接线性层时 logits 的量级被
``||w||`` 限制住（初始只有 0.0x），交叉熵会收敛得很慢。``logit_scale``（默认 16）
把 logits 拉回常见量级，等价于一个固定的余弦分类器温度，不额外引入可学习参数。

用法::

    from classification.classifier import EEGTypeClassifier

    model = EEGTypeClassifier(n_samples=1651, n_classes=37)
    model(torch.randn(8, 122, 1651))   # -> (8, 37)
"""

from __future__ import annotations

import torch
from torch import nn

from classification.model import ConvTimeAttentionEncoder


class EEGTypeClassifier(nn.Module):
    """句子级 EEG -> 语义类别的分类器。

    Parameters
    ----------
    n_samples:
        输入时间长度 ``T``（reading 2501 / imagine 1651），换 task 要重建模型。
    n_chans:
        EEG 通道数，本数据集为 122。
    n_classes:
        类别数，等于 ``ChiscoEEGTypeDataset.classes`` 的长度。
    embedding_dim:
        编码器输出维度；分类头也以它为输入宽度。
    head_dropout:
        分类头前的 dropout 概率。
    front_dropout:
        编码器前端（空间卷积之后）的 dropout 概率，透传给
        ``ConvTimeAttentionEncoder(drop_prob=...)``。
    logit_scale:
        余弦分类器的固定缩放系数，见模块文档。
    """

    def __init__(
        self,
        n_samples: int,
        n_chans: int = 122,
        n_classes: int = 39,
        embedding_dim: int = 128,
        head_dropout: float = 0.3,
        front_dropout: float = 0.3,
        logit_scale: float = 16.0,
    ):
        super().__init__()

        if n_classes < 2:
            raise ValueError("n_classes 至少为 2")
        if logit_scale <= 0:
            raise ValueError("logit_scale 必须为正")

        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.logit_scale = logit_scale

        self.encoder = ConvTimeAttentionEncoder(
            n_samples=n_samples,
            n_chans=n_chans,
            embedding_dim=embedding_dim,
            drop_prob=front_dropout,
        )
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(embedding_dim, n_classes)

    @property
    def n_time(self) -> int:
        """编码器前端输出的时间步数，用于打印。"""
        return self.encoder.n_time

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, n_classes)`` logits。"""
        embedding = self.encoder(eeg)
        return self.head(self.dropout(embedding)) * self.logit_scale
