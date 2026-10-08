"""Chisco-1.0 编码器。

EEG 编码器:
    TSConvFixedEEGEncoder
        固定长度 EEG，不使用 padding / mask / AdaptiveAvgPool。

    TSConvPointwiseEEGEncoder
        固定长度 EEG + 1×1 Pointwise Conv，压缩 TSConv feature channels。

    TSConvAdaptiveEEGEncoder
        变长 EEG，按长度分桶（Length Buckets）训练。

文本编码器:
    LaBSETextEncoder
        冻结的 LaBSE，输出 L2 归一化文本 embedding（LaBSE 为 768 维）。

用法:
    from encoder import TSConvFixedEEGEncoder, LaBSETextEncoder
"""

from .TSConvAdaptiveEEGEncoder import TSConvAdaptiveEEGEncoder
from .eeg_encoder_tsconv_fixed import TSConvFixedEEGEncoder
from .eeg_encoder_tsconv_pointwise import TSConvPointwiseEEGEncoder
from .text_encoder_labse import LaBSETextEncoder

__all__ = [
    "TSConvAdaptiveEEGEncoder",
    "TSConvFixedEEGEncoder",
    "TSConvPointwiseEEGEncoder",
    "LaBSETextEncoder",
]
