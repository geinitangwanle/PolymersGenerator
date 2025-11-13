from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


ArrayLike = Union[Sequence[float], torch.Tensor, np.ndarray]


@dataclass
class TgStats:
    """Tg 的均值/标准差与(反)标准化"""

    mean: float
    std: float

    def normalize(self, values: ArrayLike) -> np.ndarray: # 对数据集中的Tg值进行标准化
        std = self.std if self.std > 1e-6 else 1.0 # std太小时退化为 1.0，避免除零
        return (np.asarray(values, dtype=np.float32) - self.mean) / std

    def denormalize(self, values: ArrayLike) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * (self.std if self.std > 1e-6 else 1.0) + self.mean


def compute_tg_stats(values: ArrayLike) -> TgStats: # 计算 Tg 的均值和标准差
    arr = np.asarray(values, dtype=np.float32)
    return TgStats(mean=float(arr.mean()), std=float(arr.std() if arr.std() > 1e-6 else 1.0))


def _get_pad_id(tokenizer) -> int: # 获取 tokenizer 的 pad_id
    if hasattr(tokenizer, "pad_id"):
        return tokenizer.pad_id
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise AttributeError("Tokenizer must expose pad_id or pad_token_id.")


class SmilesTgDataset(Dataset):
    """支持自定义 tokenizer 及 Hugging Face tokenizer 的 Tg 数据集包装"""

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        tokenizer,
        *,
        col_smiles: str = "PSMILES", # SMILES 列名
        col_tg: str = "Tg", # Tg 列名
        max_len: int = 256,
        preprocess: Optional[Callable[[str], str]] = None, # 对原始 SMILES 的可选预处理函数
        tg_stats: Optional[TgStats] = None, # 可选的统计对象（例如包含 Tg 的均值、标准差以及归一化/反归一化方法）
    ):
        self.tokenizer = tokenizer
        self.pad_id = _get_pad_id(tokenizer)
        self.max_len = max_len
        self.preprocess = preprocess

        if isinstance(data_source, (str, Path)): # 读取数据集
            df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        else:
            raise TypeError("data_source must be CSV path or DataFrame.")

        if col_smiles not in df.columns or col_tg not in df.columns:
            raise ValueError(f"Expected columns '{col_smiles}' and '{col_tg}' in dataset.")

        self.smiles = df[col_smiles].astype(str).tolist() # 存成 Python 列表（方便后续逐条编码）
        raw_tg = df[col_tg].astype(float).to_numpy()
        self.tg_stats = tg_stats or compute_tg_stats(raw_tg)
        self.tg = self.tg_stats.normalize(raw_tg) # 标准化 Tg 值

        self.encoded = [self._encode(s) for s in self.smiles] # 预编码所有序列

    def _encode(self, smiles: str): # 编码单条 SMILES
        if self.preprocess:
            smiles = self.preprocess(smiles)
        ids = self.tokenizer.encode(
            smiles,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        return ids

    def __len__(self): # 返回数据集大小
        return len(self.encoded)

    def __getitem__(self, idx): # 返回单条数据（编码后的 SMILES 序列及对应的 Tg 值）
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.tg[idx], dtype=torch.float32)


def collate_tg(batch, pad_id: int): # 将若干样本整理成批
    token_seqs, tg_values = zip(*batch) # 聚合为两个元组
    max_len = max(seq.size(0) for seq in token_seqs) # 计算当前批次的最大序列长度
    batch_size = len(token_seqs) # 批次大小

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long) # 用 pad_id 填充
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long) # 初始化 attention_mask
    for i, seq in enumerate(token_seqs): # 填充 input_ids 和 attention_mask
        length = seq.size(0)
        input_ids[i, :length] = seq
        attention_mask[i, :length] = 1

    decoder_input_ids = input_ids[:, :-1] # decoder 输入为去掉最后一个 token 的 input_ids
    labels = input_ids[:, 1:] # labels 为去掉第一个 token 的 input_ids
    decoder_attention_mask = attention_mask[:, :-1] # decoder_attention_mask 同理

    tg_tensor = torch.stack(tg_values) # 将 Tg 值堆叠成张量

    return {
        "input_ids": input_ids, # 原始输入序列
        "attention_mask": attention_mask, # 原始输入掩码
        "decoder_input_ids": decoder_input_ids, # 用于自回归解码端的输入
        "decoder_attention_mask": decoder_attention_mask, # 用于自回归解码端的掩码
        "labels": labels, # 对齐的预测目标（下一token）
        "tg": tg_tensor, # 标准化后的 Tg 值
    }

# 把前面的 Dataset 和 DataLoader 串起来创建好
def make_loader_with_tg(
    data_source: Union[str, Path, pd.DataFrame],
    tokenizer,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    col_smiles: str = "PSMILES",
    col_tg: str = "Tg",
    max_len: int = 256,
    preprocess: Optional[Callable[[str], str]] = None,
    tg_stats: Optional[TgStats] = None,
    drop_last: bool = False,
):
    dataset = SmilesTgDataset(
        data_source,
        tokenizer,
        col_smiles=col_smiles,
        col_tg=col_tg,
        max_len=max_len,
        preprocess=preprocess,
        tg_stats=tg_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda batch: collate_tg(batch, dataset.pad_id),
    )
    return loader, dataset.tg_stats
