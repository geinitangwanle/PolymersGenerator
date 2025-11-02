import torch, torch.nn as nn
from typing import Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer # 使用 Transformer 解码器替换 GRU


try:
    from transformers import AutoModel
except ImportError:  # transformers 不是运行时必需，若未安装保持懒加载
    AutoModel = None


class VAESmiles(nn.Module):
    def __init__(
        self,
        vocab_size: int, # 必须和 tokenizer 一致（pad/bos/eos 也要对齐）
        emb_dim: int = 256, # 解码器的 embedding 维度
        encoder_hid_dim: int = 512, # RNN 编码器隐藏维度；若用 polyBERT，会改用 polyBERT.config.hidden_size
        decoder_hid_dim: Optional[int] = None, # 解码器隐藏维度，默认与 encoder_hid_dim 相同
        z_dim: int = 128, # 潜空间维度
        n_layers: int = 1, # 编码器/解码器的 GRU 层数
        pad_id: int = 0, # 填充 token id
        bos_id: int = 1, # 序列起始 token id
        eos_id: int = 2, # 序列结束 token id
        drop: float = 0.1, # dropout 比例
        use_polybert: bool = False, # 是否使用预训练 polyBERT 替代 RNN 编码器
        polybert_name: str = "kuelumbus/polyBERT", # 导入预训练polyBERT
        polybert: Optional[nn.Module] = None, # 可传入自定义 polyBERT 模型实例
        freeze_polybert: bool = True, # 是否冻结 polyBERT 参数
        polybert_pooling: str = "cls", # "cls" 或 "mean" 池化句向量
        max_len = 256, # 用于位置编码的最大序列长度
    ):
        super().__init__()
        decoder_hid_dim = decoder_hid_dim or encoder_hid_dim

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.drop = nn.Dropout(drop)
        # 仅用于解码器输入（teacher forcing / 采样时）；padding_idx 让 pad 的梯度为0
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id) 

        self.use_polybert = use_polybert
        self.polybert_pooling = polybert_pooling
        self.max_len = max_len

        # 位置编码（可学习）
        self.pos_emb = nn.Embedding(self.max_len, emb_dim)

        # 把 z 投影到解码器维度 E
        self.latent_proj = nn.Linear(z_dim, emb_dim)

        # Transformer 解码器层
        decoder_layer = TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=8,
            dim_feedforward=4 * emb_dim,
            dropout=drop,
            batch_first=True,   # 让输入输出都是 [B, T, E]
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=4)

        # 输出层（可选权重共享）
        self.out = nn.Linear(emb_dim, vocab_size)

        if use_polybert:
            if polybert is not None:
                self.polybert = polybert
            else:
                if AutoModel is None:
                    raise ImportError("transformers is required when use_polybert=True")
                self.polybert = AutoModel.from_pretrained(polybert_name)
            if freeze_polybert and self.polybert is not None:
                self.polybert.eval()
            self.encoder = None # 不使用 RNN 编码器
            self.encoder_hidden_dim = getattr(self.polybert.config, "hidden_size", encoder_hid_dim) # polyBERT 输出维度
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

        self.mu = nn.Linear(self.encoder_hidden_dim, z_dim) # 均值层
        self.logvar = nn.Linear(self.encoder_hidden_dim, z_dim) # 对数方差层

        self.decoder_hidden_dim = decoder_hid_dim # 解码器隐藏维度
        


    def encode(self, x, attention_mask=None):
        # x: [B, T]，批量的 token id 序列
        if self.use_polybert: # 如果用 polyBERT，需要告诉模型哪些位置是有效 token，哪些是 padding
            if attention_mask is None:
                attention_mask = (x != self.pad_id).long() # 自动生成掩码：pad 位置（pad_id）为 0，其他为 1
            outputs = self.polybert(input_ids=x, attention_mask=attention_mask) # 输出 Transformer 各层最后一层隐藏状态
            hidden = outputs.last_hidden_state  # [B, T, H]
            if self.polybert_pooling == "cls": # cls 池化
                pooled = hidden[:, 0] # 取第一个 token（通常是 [CLS]）的隐状态，代表整个序列语义
            else:
                mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
                summed = (hidden * mask).sum(dim=1) # 按 mask 求和
                denom = mask.sum(dim=1).clamp(min=1) # 防止除以0
                pooled = summed / denom # [B, H]
            mu, logvar = self.mu(pooled), self.logvar(pooled) # 两个线性层 nn.Linear(H, z_dim) 把句向量映射到潜空间的均值和对数方差
            return mu, logvar

        # 使用 RNN 编码器
        emb = self.drop(self.emb(x))              # [B,T,E]
        _, h = self.encoder(emb)                  # h: [L,B,H]
        h_last = h[-1]                            # [B,H]
        mu, logvar = self.mu(h_last), self.logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar): # 重参数化（保持可导）
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher_forcing(self, z, x_inp):
        """
        z: [B, Z]
        x_inp: [B, T]  （teacher forcing 输入，通常是右移后的序列）
        返回 logits: [B, T, V]
        """
        B, T = x_inp.size()
        device = x_inp.device

        # 1) 将 嵌入 + 位置编码 作为解码器输入
        pos_ids = torch.arange(T, device=device).unsqueeze(0)           # [1, T]
        tgt = self.emb(x_inp) + self.pos_emb(pos_ids)                   # [B, T, E]
        tgt = self.drop(tgt)

        # 2) 构造 causal mask（阻止看未来）
        tgt_mask = self._causal_mask(T, device=device)                  # [T, T]

        # 3) 构造 memory：把 z 投影到 E，然后扩展到每个时间步
        cond = self.latent_proj(z).unsqueeze(1)                         # [B, 1, E]
        memory = cond.expand(B, T, cond.size(-1))                       # [B, T, E]

        # 4）padding mask（对 tgt 的 PAD 做 key_padding_mask）
        tgt_key_padding_mask = (x_inp == self.pad_id)                   # [B, T], True=要屏蔽

        
        # 5) Transformer 解码
        y = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,  # 屏蔽解码器输入的 pad
        )                                              # [B, T, E]

        # 6) 输出层
        logits = self.out(self.drop(y))                                   # [B, T, V]
        return logits
    
    # 把“编码→重参数化→teacher forcing 解码”串起来，返回训练所需的三样：logits, mu, logvar
    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None):
        mu, logvar = self.encode(encoder_input_ids, encoder_attention_mask) # 编码
        z = self.reparameterize(mu, logvar)                     # 重参数化
        logits = self.decode_teacher_forcing(z, decoder_input_ids) # 解码
        return logits, mu, logvar

    # 推理/生成阶段，从给定潜向量 z 自回归生成序列（无 teacher forcing）
    @torch.no_grad()
    def sample(self, z, max_len=256, bos_id=None, eos_id=None, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
        """
        从潜变量 z 自回归生成序列（Transformer 解码器）。
        支持温度 / top-k / top-p（可选），默认贪婪 ~ temperature=1, 无 top-k/p。
        """
        device = z.device
        B = z.size(0)
        bos_tok = self.bos_id if bos_id is None else bos_id
        eos_tok = self.eos_id if eos_id is None else eos_id

        # 条件 memory（不随步长变化）
        cond = self.latent_proj(z)                      # [B, E]

        # 当前序列，起始为 BOS
        cur = torch.full((B, 1), bos_tok, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # 自回归生成
        for t in range(1, max_len):
            pos_ids = torch.arange(cur.size(1), device=device).unsqueeze(0)   # [1, T]
            tgt = self.emb(cur) + self.pos_emb(pos_ids)                       # [B, T, E]
            tgt_mask = self._causal_mask(cur.size(1), device=device)
            memory = cond.unsqueeze(1).expand(B, cur.size(1), cond.size(-1))  # [B, T, E]

            y = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(cur == self.pad_id)  # 可选；BOS/EOS 一般不是 PAD
            )                                              # [B, T, E]

            logits = self.out(y[:, -1:, :])                # 只取最后一步 [B,1,V]
            if temperature != 1.0:
                logits = logits / temperature

            # 采样策略：贪婪 / top-k / top-p
            probs = torch.softmax(logits.squeeze(1), dim=-1)  # [B, V]

            if top_k is not None:
                topk_vals, topk_idx = probs.topk(top_k, dim=-1)
                probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
                next_local = torch.multinomial(probs, 1)
                next_tok = topk_idx.gather(-1, next_local)
            elif top_p is not None:
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum > top_p
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_local = torch.multinomial(sorted_probs, 1)
                next_tok = sorted_idx.gather(-1, next_local)
            else:
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)  # 贪婪

            # 逐样本结束
            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, eos_tok), next_tok)
            finished |= (next_tok.squeeze(1) == eos_tok)

            cur = torch.cat([cur, next_tok], dim=1)
            if finished.all():
                break

        return cur  # [B, T_gen]，以 BOS 开头，遇到 EOS 可在外部截断
            
    def _causal_mask(self, T: int, device):
    # 下三角为 False（可见），上三角为 True（屏蔽）
    # PyTorch 的 Transformer 期望 True 表示要 mask 的位置
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
