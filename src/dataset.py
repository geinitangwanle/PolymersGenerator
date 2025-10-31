import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional
from pathlib import Path


def _get_pad_id(tokenizer) -> int: # 获取 tokenizer 的 pad_id
    if hasattr(tokenizer, "pad_id"):
        return tokenizer.pad_id
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise AttributeError("Tokenizer must expose `pad_id` or `pad_token_id`.")


class SmilesDataset(Dataset):
    """支持自定义 tokenizer 及 Hugging Face tokenizer 的简单数据集包装。"""

    def __init__(
        self,
        data_source, # 可以是 CSV 路径、pandas DataFrame、或字符串序列
        tokenizer, # 需提供 encode()，以及 pad_id 或 pad_token_id
        col: str = "PSMILES",
        max_len: int = 256, # 最终 token 序列的最大长度（包含特殊符号）
        preprocess: Optional[Callable[[str], str]] = None, # 可选的字符串级预处理（如标准化 SMILES、去空格、大小写统一等）
    ):
        self.tokenizer = tokenizer
        self.pad_id = _get_pad_id(tokenizer)
        self.max_len = max_len
        self.preprocess = preprocess

        if isinstance(data_source, (str, Path)): # csv
            df = pd.read_csv(data_source)
            seq_iter = df[col].astype(str).tolist()
        elif isinstance(data_source, pd.DataFrame): # dataframe
            seq_iter = data_source[col].astype(str).tolist()
        else: # 序列
            seq_iter = [str(s) for s in data_source]

        self.seqs = [self._encode(s) for s in seq_iter] # 预编码所有序列

    def _encode(self, smiles: str):
        if self.preprocess: # 如果有预处理函数，则先处理
            smiles = self.preprocess(smiles)

        if hasattr(self.tokenizer, "bos_id"): # 自定义 tokenizer: 不自动添加特殊 token
            ids = self.tokenizer.encode(smiles)
            return ids[: self.max_len]

        # Hugging Face tokenizer: 自动添加特殊 token
        ids = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        return ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.long)


def collate_pad(batch, pad_id): # 对 batch 内的序列进行 padding，并生成相应的 attention mask
    max_len = max(x.size(0) for x in batch) # 找出 batch 内最长序列的长度
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long) # 用 pad_id 填充
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long) # 初始化 attention mask
    for i, x in enumerate(batch):
        seq_len = x.size(0)
        input_ids[i, :seq_len] = x
        attention_mask[i, :seq_len] = 1
    decoder_input_ids = input_ids[:, :-1]  # 去掉最后一个位置（形状 [B, max_len-1]）→ 作为“当前步输入”
    labels = input_ids[:, 1:]  # 去掉第一个位置（形状同上）→ 作为“下一步目标”
    decoder_attention_mask = attention_mask[:, :-1] # 与 decoder_input_ids 对齐的注意力掩码
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }


class SmilesCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        return collate_pad(batch, self.pad_id)


def make_loader(
    data_source,
    tokenizer,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    col: str = "smiles",
    max_len: int = 256,
    preprocess: Optional[Callable[[str], str]] = None,
):
    dataset = SmilesDataset(
        data_source,
        tokenizer,
        col=col,
        max_len=max_len,
        preprocess=preprocess,
    ) # 创建数据集实例
    pad_id = _get_pad_id(tokenizer) # 获取 pad_id
    return DataLoader( # 创建数据加载器
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SmilesCollator(pad_id),
    )
