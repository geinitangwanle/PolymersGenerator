import torch, torch.nn as nn
from typing import Optional

try:
    from transformers import AutoModel
except ImportError:  # transformers 不是运行时必需，若未安装保持懒加载
    AutoModel = None


class VAESmiles(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        encoder_hid_dim: int = 512,
        decoder_hid_dim: Optional[int] = None,
        z_dim: int = 128,
        n_layers: int = 1,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        drop: float = 0.1,
        use_polybert: bool = False,
        polybert_name: str = "kuelumbus/polyBERT",
        polybert: Optional[nn.Module] = None,
        freeze_polybert: bool = True,
        polybert_pooling: str = "cls",
    ):
        """
        Args:
            vocab_size: 词表大小，需与 tokenizer 对齐。
            emb_dim: 解码器 embedding 维度。
            encoder_hid_dim: 若不使用 polyBERT 时的编码器隐藏维度。
            decoder_hid_dim: 解码器隐藏层维度，默认与 encoder_hid_dim 相同。
            z_dim: 潜变量维度。
            n_layers: 解码器/编码器层数（GRU）。
            pad_id/bos_id/eos_id: 关键特殊 token id。
            use_polybert: 是否以预训练 polyBERT 取代 RNN 编码器。
            polybert_name: AutoModel 名称。
            polybert: 预先构造好的 polyBERT 模型，可复用外部实例。
            freeze_polybert: 是否在训练时冻结 polyBERT。
            polybert_pooling: "cls" 或 "mean"，用于从编码器输出得到句向量。
        """
        super().__init__()
        decoder_hid_dim = decoder_hid_dim or encoder_hid_dim

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.drop = nn.Dropout(drop)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        self.use_polybert = use_polybert
        self.polybert_pooling = polybert_pooling

        if use_polybert:
            if polybert is not None:
                self.polybert = polybert
            else:
                if AutoModel is None:
                    raise ImportError("transformers is required when use_polybert=True")
                self.polybert = AutoModel.from_pretrained(polybert_name)
            if freeze_polybert:
                for p in self.polybert.parameters():
                    p.requires_grad = False
            self.encoder = None
            self.encoder_hidden_dim = getattr(self.polybert.config, "hidden_size", encoder_hid_dim)
        else:
            self.polybert = None
            self.encoder_hidden_dim = encoder_hid_dim
            self.encoder = nn.GRU(
                emb_dim,
                self.encoder_hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=False,
            )

        self.mu = nn.Linear(self.encoder_hidden_dim, z_dim)
        self.logvar = nn.Linear(self.encoder_hidden_dim, z_dim)

        self.decoder_hidden_dim = decoder_hid_dim
        self.n_layers = n_layers
        self.z_to_h = nn.Linear(z_dim, self.decoder_hidden_dim * n_layers)
        self.decoder = nn.GRU(
            emb_dim,
            self.decoder_hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.out = nn.Linear(self.decoder_hidden_dim, vocab_size)

    def encode(self, x, attention_mask=None):
        # x: [B, T]
        if self.use_polybert:
            if attention_mask is None:
                attention_mask = (x != self.pad_id).long()
            outputs = self.polybert(input_ids=x, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # [B, T, H]
            if self.polybert_pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = summed / denom
            mu, logvar = self.mu(pooled), self.logvar(pooled)
            return mu, logvar

        emb = self.drop(self.emb(x))              # [B,T,E]
        _, h = self.encoder(emb)                  # h: [L,B,H]
        h_last = h[-1]                            # [B,H]
        mu, logvar = self.mu(h_last), self.logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher_forcing(self, z, x_inp):
        # z: [B,Z]; x_inp: [B,T] (输入序列, 已对齐到 t-1)
        B = z.size(0)
        init_h = (
            self.z_to_h(z)
            .view(self.n_layers, B, self.decoder_hidden_dim)
            .contiguous()
        )
        emb = self.drop(self.emb(x_inp))
        y, _ = self.decoder(emb, init_h)
        logits = self.out(self.drop(y))          # [B,T,V]
        return logits

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None):
        mu, logvar = self.encode(encoder_input_ids, encoder_attention_mask)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_teacher_forcing(z, decoder_input_ids)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, z, max_len=256, bos_id=None, eos_id=None):
        B = z.size(0)
        h = (
            self.z_to_h(z)
            .view(self.n_layers, B, self.decoder_hidden_dim)
            .contiguous()
        )
        bos_tok = self.bos_id if bos_id is None else bos_id
        eos_tok = self.eos_id if eos_id is None else eos_id
        # 初始 token
        cur = torch.full((B,1), bos_tok, dtype=torch.long, device=z.device)
        outputs = [cur]
        for _ in range(max_len-1):
            emb = self.emb(cur)                  # [B,1,E]
            y, h = self.decoder(emb, h)         # [B,1,H]
            logits = self.out(y)                # [B,1,V]
            next_tok = torch.argmax(logits, dim=-1)  # 贪婪（可换为温度/采样）
            outputs.append(next_tok)
            cur = next_tok
            if (next_tok == eos_tok).all():
                break
        return torch.cat(outputs, dim=1)        # [B,T]
