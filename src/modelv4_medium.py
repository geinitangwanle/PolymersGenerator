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
        # 若 checkpoint 已包含这些超参，优先使用；否则落到中号默认值
        emb_dim = kwargs.pop("emb_dim", 384)
        decoder_hid_dim = kwargs.pop("decoder_hid_dim", 768)
        z_dim = kwargs.pop("z_dim", 256)
        cond_latent_dim = kwargs.pop("cond_latent_dim", 64)
        tg_hidden_dim = kwargs.pop("tg_hidden_dim", 256)
        num_decoder_layers = kwargs.pop("num_decoder_layers", 6)
        decoder_nhead = kwargs.pop("decoder_nhead", 12)  # 384 / 12 = 32
        decoder_ff_mult = kwargs.pop("decoder_ff_mult", 4)
        super().__init__(
            vocab_size,
            emb_dim=emb_dim,
            decoder_hid_dim=decoder_hid_dim,
            z_dim=z_dim,
            cond_latent_dim=cond_latent_dim,
            tg_hidden_dim=tg_hidden_dim,
            num_decoder_layers=num_decoder_layers,
            decoder_nhead=decoder_nhead,
            decoder_ff_mult=decoder_ff_mult,
            **kwargs,
        )
