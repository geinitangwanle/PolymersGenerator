from typing import Optional
from transformers import AutoTokenizer


class PolyBertTokenizer:

    def __init__(self, name: str = "kuelumbus/polyBERT", tokenizer=None, use_fast: bool = False):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else: # 默认使用polyBERT的分词器
            self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast) # use_fast=True时使用fast tokenizer

        # Ensure pad token exists; fall back to SEP/EOS if needed.
        if self.tokenizer.pad_token is None: # 优先使用已有的 pad_token,没有的话就回退到 SEP 或 EOS 作为 pad。
            fallback = self.tokenizer.sep_token or self.tokenizer.eos_token
            if fallback is None:
                raise ValueError("Tokenizer must define pad or sep/eos tokens.")
            pad_id = self.tokenizer.convert_tokens_to_ids(fallback) # 获取回退标记的ID
            if pad_id == self.tokenizer.unk_token_id: # 如果回退标记是未知标记，则添加一个新的填充标记
                self.tokenizer.add_special_tokens({"pad_token": fallback})
            else:
                self.tokenizer.pad_token = fallback # 否则直接设置回退标记为填充标记

    @property
    def pad_id(self) -> int: # 统一对外暴露 pad id，避免上层代码反复从内部对象取。
        return self.tokenizer.pad_token_id

    @property
    def bos_id(self) -> Optional[int]:  # 优先返回 CLS；否则退回 BOS
        # Prefer CLS, otherwise fall back to BOS if available.
        if self.tokenizer.cls_token_id is not None:
            return self.tokenizer.cls_token_id
        return getattr(self.tokenizer, "bos_token_id", None)

    @property
    def eos_id(self) -> Optional[int]: # 优先用 SEP；否则退回 EOS
        if self.tokenizer.sep_token_id is not None:
            return self.tokenizer.sep_token_id
        return getattr(self.tokenizer, "eos_token_id", None)

    @property
    def vocab_size(self) -> int:
        base = self.tokenizer.vocab_size
        if hasattr(self.tokenizer, "get_added_vocab"): # 考虑到可能有新增的词表
            base += len(self.tokenizer.get_added_vocab()) # 实际可用的总词表大小
        return base

    def encode(self, text: str):
        return self.tokenizer.encode(
            text,
            add_special_tokens=True, # 自动在两端加入[CLS] ... [SEP]
            truncation=False,  # 让上层决定截断
        )

    def decode(self, ids, skip_special_tokens: bool = True): # 解码时默认跳过特殊标记
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self): # 暴露底层词表
        return self.tokenizer.get_vocab()
