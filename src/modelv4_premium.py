"""
高配版的 ConditionalVAESmiles（基于 modelv4），进一步放大容量：
- 更宽的 embedding/潜变量/FiLM 维度
- 更多的 Transformer 解码层、注意力头与前馈扩张
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
            emb_dim=512,
            decoder_hid_dim=1024,
            z_dim=384,
            cond_latent_dim=128,
            tg_hidden_dim=512,
            num_decoder_layers=8,
            decoder_nhead=16,  # 512 / 16 = 32
            decoder_ff_mult=8,  # 更宽的前馈层
            **kwargs,
        )
