import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional
from pathlib import Path


def _get_pad_id(tokenizer) -> int:
    if hasattr(tokenizer, "pad_id"):
        return tokenizer.pad_id
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise AttributeError("Tokenizer must expose `pad_id` or `pad_token_id`.")


class SmilesDataset(Dataset):
    """支持自定义 tokenizer 及 Hugging Face tokenizer 的简单数据集包装。"""

    def __init__(
        self,
        data_source,
        tokenizer,
        col: str = "smiles",
        max_len: int = 256,
        preprocess: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            data_source: 可以是 CSV 路径、pandas DataFrame、或字符串序列。
            tokenizer: 需提供 encode()，以及 pad_id 或 pad_token_id。
            col: 当 data_source 为 CSV/DataFrame 时使用的列名。
            max_len: token 序列的最大长度（含特殊符号）。
            preprocess: 可选的字符串预处理函数。
        """
        self.tokenizer = tokenizer
        self.pad_id = _get_pad_id(tokenizer)
        self.max_len = max_len
        self.preprocess = preprocess

        if isinstance(data_source, (str, Path)):
            df = pd.read_csv(data_source)
            seq_iter = df[col].astype(str).tolist()
        elif isinstance(data_source, pd.DataFrame):
            seq_iter = data_source[col].astype(str).tolist()
        else:
            seq_iter = [str(s) for s in data_source]

        self.seqs = [self._encode(s) for s in seq_iter]

    def _encode(self, smiles: str):
        if self.preprocess:
            smiles = self.preprocess(smiles)

        if hasattr(self.tokenizer, "bos_id"):
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


def collate_pad(batch, pad_id):
    max_len = max(x.size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        seq_len = x.size(0)
        input_ids[i, :seq_len] = x
        attention_mask[i, :seq_len] = 1
    decoder_input_ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    decoder_attention_mask = attention_mask[:, :-1]
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
    )
    pad_id = _get_pad_id(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SmilesCollator(pad_id),
    )
