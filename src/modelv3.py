import torch
import torch.nn as nn
from typing import Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None


class ConditionalVAESmiles(nn.Module):
    """生成式的条件变分自编码器模型，用于分子SMILES字符串的生成，使用了Tg作为条件输入。"""

    def __init__(
        self,
        vocab_size: int, # 词表大小（输出分类维度）
        *,
        emb_dim: int = 256, # token/位置嵌入维度
        encoder_hid_dim: int = 512, # 自建编码器隐藏维度（不用 polyBERT 时有效）
        decoder_hid_dim: Optional[int] = None, 
        z_dim: int = 128, # VAE 潜变量维度
        cond_dim: int = 1, # 条件向量输入维度（例如 Tg=1）
        cond_latent_dim: int = 32, # 条件向量（Tg）经 MLP 映射后的维度
        pad_id: int = 0, # padding token ID
        bos_id: int = 1, # begin-of-sequence token ID
        eos_id: int = 2, # end-of-sequence token ID
        drop: float = 0.1, # 暂退层概率
        use_polybert: bool = True, # 是否使用 polyBERT 作为编码器
        polybert_name: str = "kuelumbus/polyBERT",
        polybert: Optional[nn.Module] = None,
        freeze_polybert: bool = False, # 是否冻结 polyBERT 参数
        polybert_pooling: str = "cls", # 'cls' or 'mean' 池化方式
        max_len: int = 256, # 最大序列长度（位置嵌入用）
        use_tg_regression: bool = True, # 是否使用 Tg 回归头
        tg_hidden_dim: int = 128, # Tg 回归头隐藏层维度
    ):
        super().__init__()
        decoder_hid_dim = decoder_hid_dim or encoder_hid_dim

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.drop = nn.Dropout(drop)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.max_len = max_len

        self.use_polybert = use_polybert
        self.polybert_pooling = polybert_pooling

        if use_polybert:
            if polybert is not None:
                self.polybert = polybert # 传入预训练的polyBert模型
            else:
                if AutoModel is None:
                    raise ImportError("transformers is required when use_polybert=True")
                self.polybert = AutoModel.from_pretrained(polybert_name)
            if freeze_polybert and self.polybert is not None:
                self.polybert.eval()
            self.encoder_hidden_dim = getattr(self.polybert.config, "hidden_size", encoder_hid_dim)
            self.encoder = None
        else:
            self.polybert = None
            self.encoder_hidden_dim = encoder_hid_dim
            self.encoder = nn.GRU(
                emb_dim,
                self.encoder_hidden_dim,
                num_layers=1,
                batch_first=True,
            )

        self.mu = nn.Linear(self.encoder_hidden_dim, z_dim) # VAE 潜变量均值层
        self.logvar = nn.Linear(self.encoder_hidden_dim, z_dim) # VAE 潜变量对数方差层

        self.cond_encoder = nn.Sequential( # 将条件向量（Tg）映射到潜变量空间
            nn.Linear(cond_dim, cond_latent_dim), # 线性层：1 -> cond_latent_dim
            nn.SiLU(), # SiLU 激活函数
            nn.Linear(cond_latent_dim, cond_latent_dim),# 线性层：cond_latent_dim -> cond_latent_dim
        )
        self.cond_latent_dim = cond_latent_dim # 条件潜变量维度
        self.z_dim = z_dim # 基础潜变量维度

        self.pos_emb = nn.Embedding(self.max_len, emb_dim) # 位置嵌入层
        self.latent_proj = nn.Linear(z_dim + cond_latent_dim, emb_dim) #将拼接后的潜变量映射到解码器输入维度

        decoder_layer = TransformerDecoderLayer( # Transformer 解码器层
            d_model=emb_dim, # 嵌入维度
            nhead=8, # 注意力头数
            dim_feedforward=4 * emb_dim, # 前馈网络维度 4×d_model
            dropout=drop, # 暂退概率
            batch_first=True, # 输入输出张量形状为 (batch, seq, feature）
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=4) # 堆叠4层解码器层
        self.out = nn.Linear(emb_dim, vocab_size) # 输出层，映射到词表大小

        self.use_tg_regression = use_tg_regression # 是否使用 Tg 回归头，训练时可以联合目标：重构（LM）+ KL + Tg 回归
        if use_tg_regression:
            self.tg_head = nn.Sequential(
                nn.Linear(z_dim, tg_hidden_dim),
                nn.SiLU(), # SiLU 激活函数
                nn.Linear(tg_hidden_dim, 1),
            )
        else:
            self.tg_head = None # 不使用 Tg 回归头

    def encode(self, x, attention_mask=None):  # 编码器，将输入序列编码为潜变量的均值和对数方差
        if self.use_polybert and self.polybert is not None:
            if attention_mask is None:
                attention_mask = (x != self.pad_id).long()
            outputs = self.polybert(input_ids=x, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state # 调用 HF 模型得到最后一层隐表示 hidden，形状 [B, T, H].
            if self.polybert_pooling == "cls": # 若用 CLS 池化：取第一个位置向量 [B, H]
                pooled = hidden[:, 0]
            else: # 若用均值池化：对非 padding 部分求均值 [B, H]
                mask = attention_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = summed / denom
            return self.mu(pooled), self.logvar(pooled)

        emb = self.drop(self.emb(x))
        _, h = self.encoder(emb)
        h_last = h[-1]
        return self.mu(h_last), self.logvar(h_last)

    def reparameterize(self, mu, logvar): # 重参数化技巧，从均值和对数方差采样潜变量
        """标准 VAE 采样：z = mu + eps * std，其中 std = exp(0.5*logvar)；eps ~ N(0, I)"""
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)
        return mu + eps * std

    def _prepare_latent(self, z_base, conditions): # 在前向传播中拼接基础潜变量和条件潜变量
        cond_latent = self.cond_encoder(conditions)
        return torch.cat([z_base, cond_latent], dim=-1)

    def decode_teacher_forcing(self, z_concat, x_inp): # 解码器，使用教师强制（teacher forcing）进行训练
        B, T = x_inp.size()
        device = x_inp.device

        pos_ids = torch.arange(T, device=device).unsqueeze(0) # 位置ID张量 [1, T]
        tgt = self.emb(x_inp) + self.pos_emb(pos_ids) # 解码器输入嵌入 + 位置嵌入 [B, T, emb_dim]
        tgt = self.drop(tgt) # 应用暂退

        tgt_mask = self._causal_mask(T, device=device) # 因果掩码，防止解码器看到未来信息
        memory = self.latent_proj(z_concat).unsqueeze(1).expand(B, T, -1) # 将潜变量映射并扩展为解码器记忆张量 [B, T, emb_dim]
        tgt_key_padding_mask = (x_inp == self.pad_id) # 解码器输入的 padding 掩码 [B, T]

        y = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.out(self.drop(y))
        return logits

    def forward(self, encoder_input_ids, decoder_input_ids, conditions, encoder_attention_mask=None):
        mu, logvar = self.encode(encoder_input_ids, encoder_attention_mask) # 编码器得到潜变量均值和对数方差
        z_base = self.reparameterize(mu, logvar) # 重参数化采样基础潜变量
        z_concat = self._prepare_latent(z_base, conditions) # 拼接基础潜变量和条件潜变量
        logits = self.decode_teacher_forcing(z_concat, decoder_input_ids) # 解码器得到输出 logits
        tg_pred = self.tg_head(z_base) if self.tg_head is not None else None
        return logits, mu, logvar, tg_pred # 返回解码器输出 logits，潜变量均值和对数方差，以及 Tg 预测（如有）

    @torch.no_grad()
    def sample(
        self,
        num_samples: int, # 生成样本数量
        conditions: torch.Tensor,
        *,
        max_len: int = 256,
        temperature: float = 1.0, # 采样温度
        top_k: Optional[int] = None, # top-k 采样参数
        top_p: Optional[float] = None, # top-p 采样参数
        z_base: Optional[torch.Tensor] = None,
    ):
        device = conditions.device
        if z_base is None: # 若未提供基础潜变量，则随机采样
            z_base = torch.randn(num_samples, self.z_dim, device=device) 
        cond_latent = self.cond_encoder(conditions)
        z_concat = torch.cat([z_base, cond_latent], dim=-1)

        B = z_concat.size(0)
        cur = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        memory = self.latent_proj(z_concat)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1): # 自回归逐步生成序列
            pos_ids = torch.arange(cur.size(1), device=device).unsqueeze(0)
            tgt = self.emb(cur) + self.pos_emb(pos_ids)
            tgt_mask = self._causal_mask(cur.size(1), device=device)

            y = self.decoder(
                tgt=tgt,
                memory=memory.unsqueeze(1).expand(B, cur.size(1), -1),
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(cur == self.pad_id),
            )
            logits = self.out(y[:, -1])
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_k is not None:
                probs = self._top_k(probs, top_k)
            elif top_p is not None:
                probs = self._top_p(probs, top_p)

            next_tok = torch.multinomial(probs, num_samples=1)
            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, self.eos_id), next_tok)
            finished |= next_tok.squeeze(1) == self.eos_id
            cur = torch.cat([cur, next_tok], dim=1)
            if finished.all():
                break

        return cur

    def _top_k(self, probs, k):
        topk_vals, topk_idx = probs.topk(k, dim=-1)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, topk_idx, topk_vals)
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        return filtered

    def _top_p(self, probs, top_p):
        sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(1, sorted_idx, sorted_probs)
        return filtered

    def _causal_mask(self, T: int, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
