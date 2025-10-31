import torch, torch.nn as nn
from typing import Optional

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
        self.n_layers = n_layers # 解码器层数
        self.z_to_h = nn.Linear(z_dim, self.decoder_hidden_dim * n_layers) # 从 z 到解码器初始隐藏状态的映射
        self.decoder = nn.GRU(
            emb_dim,
            self.decoder_hidden_dim,
            num_layers=n_layers,
            batch_first=True, 
        )
        self.out = nn.Linear(self.decoder_hidden_dim, vocab_size)

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
        # z: [B,Z]; x_inp: [B,T] (输入序列, 已对齐到 t-1)
        # x_inp: [B, T] 教师信号序列（一般是目标序列右移一位：BOS, y0, y1, ...）
        B = z.size(0) # batch size

        # (1) 把 z 映射为解码器各层初始隐状态 h0
        init_h = (
            self.z_to_h(z) # [B, L*H_dec]
            .view(self.n_layers, B, self.decoder_hidden_dim) # [L, B, H_dec]
            .contiguous() # 保持内存连续性
        )

        # (2) 把 teacher-forcing 输入做嵌入 + dropout
        emb = self.drop(self.emb(x_inp))        # [B, T, E]
        # (3) 以 init_h 为初始状态，整段送入 GRU
        y, _ = self.decoder(emb, init_h)        # y: [B, T, H_dec]
        # (4) 线性投影到词表维度，得到每步的 logits
        logits = self.out(self.drop(y))         # [B,T,V]
        return logits
    
    # 把“编码→重参数化→teacher forcing 解码”串起来，返回训练所需的三样：logits, mu, logvar
    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask=None):
        mu, logvar = self.encode(encoder_input_ids, encoder_attention_mask) # 编码
        z = self.reparameterize(mu, logvar)                     # 重参数化
        logits = self.decode_teacher_forcing(z, decoder_input_ids) # 解码
        return logits, mu, logvar

    # 推理/生成阶段，从给定潜向量 z 自回归生成序列（无 teacher forcing）
    @torch.no_grad()
    def sample(self, z, max_len=256, bos_id=None, eos_id=None):
        B = z.size(0)
        h = (
            self.z_to_h(z) # 把一个向量 z 展开成每一层 GRU 的初始隐状态
            .view(self.n_layers, B, self.decoder_hidden_dim) # [L, B, H_dec]
            .contiguous() # 保持内存连续性
        )
        bos_tok = self.bos_id if bos_id is None else bos_id # 起始 token
        eos_tok = self.eos_id if eos_id is None else eos_id # 结束 token
        # 初始 token
        cur = torch.full((B,1), bos_tok, dtype=torch.long, device=z.device) #当前时间步要输入解码器的 token（初始为 BOS），形状 [B, 1]
        outputs = [cur] # 保存已生成的 token 序列列表，先把 BOS 放进去
        for _ in range(max_len-1): # 自回归循环（重复生成下一 token）
            emb = self.emb(cur)                  # [B,1,E]
            y, h = self.decoder(emb, h)         # [B,1,H]
            logits = self.out(y)                # [B,1,V]
            next_tok = torch.argmax(logits, dim=-1)  # 贪婪（可换为温度/采样）
            outputs.append(next_tok)
            cur = next_tok
            if (next_tok == eos_tok).all():
                break
        return torch.cat(outputs, dim=1)        # [B,T] 拼接所有时间步的输出 token
