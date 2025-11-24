"""
中号配置的 ConditionalVAESmiles（基于 modelv4），相比基础版有更高的容量：
- 更宽的 embedding/潜变量/FiLM 维度
- 更多的 Transformer 解码层与注意力头
"""

try:
    from modelv4 import ConditionalVAESmiles as _BaseConditionalVAESmiles
except ImportError:  # 兼容作为包导入
    from .modelv4 import ConditionalVAESmiles as _BaseConditionalVAESmiles

__all__ = ["ConditionalVAESmiles"]


class ConditionalVAESmiles(_BaseConditionalVAESmiles):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(
            vocab_size,
            emb_dim=384,
            decoder_hid_dim=768,
            z_dim=256,
            cond_latent_dim=64,
            tg_hidden_dim=256,
            num_decoder_layers=6,
            decoder_nhead=12,  # 384 / 12 = 32
            decoder_ff_mult=4,
            **kwargs,
        )
