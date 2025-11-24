import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoModel

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_ROOT.parent.parent  # .../PolymersGenerator
sys.path.append(str(PROJ_ROOT / "src"))

from dataset_tg import make_loader_with_tg
from modelv3 import ConditionalVAESmiles
from tokenizer import PolyBertTokenizer
from train import (
    configure_polybert_finetuning,
    kld_loss,
    set_seed,
    split_dataframe,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tg-conditional polymer VAE.")
    parser.add_argument("--csv", type=Path, default=Path("data/PSMILES_Tg_only.csv")) # 输入数据 CSV 文件路径
    parser.add_argument("--col-smiles", type=str, default="PSMILES") # SMILES 列名
    parser.add_argument("--col-tg", type=str, default="Tg") # Tg 列名
    parser.add_argument("--polybert-dir", type=Path, default=Path("./polybert")) # polyBERT 模型目录
    parser.add_argument("--epochs", type=int, default=20) # 训练轮数
    parser.add_argument("--batch-size", type=int, default=128) # 批大小
    parser.add_argument("--max-len", type=int, default=256) # 最大序列长度
    parser.add_argument("--lr", type=float, default=3e-4) # 基础学习率
    parser.add_argument("--polybert-lr", type=float, default=1e-5) # polyBERT 学习率
    parser.add_argument("--polybert-train-last-n", type=int, default=0) # 微调 polyBERT 的最后 N 层
    parser.add_argument("--weight-decay", type=float, default=0.01) # 权重衰减系数
    parser.add_argument("--kl-warmup", type=int, default=10) # KL 散度权重预热轮数
    parser.add_argument("--lambda-tg", type=float, default=0.1) # Tg 回归损失权重
    parser.add_argument("--seed", type=int, default=42)    # 随机种子
    parser.add_argument("--output", type=Path, default=Path("checkpoints/modelv3_tg.pt")) # 模型检查点保存路径
    parser.add_argument("--num-workers", type=int, default=4) # 数据加载器工作线程数
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/mps/cpu).") # 设备覆盖选项
    parser.add_argument("--emb-dim", type=int, default=256) # 嵌入维度
    parser.add_argument("--decoder-hid-dim", type=int, default=512) # 解码器隐藏层维度
    parser.add_argument("--z-dim", type=int, default=128) # 潜变量维度
    parser.add_argument("--cond-latent-dim", type=int, default=32) # 条件潜变量维度
    parser.add_argument("--tg-hidden-dim", type=int, default=128) # Tg 回归头隐藏层维度
    parser.add_argument("--dropout", type=float, default=0.1) # 暂退概率
    parser.add_argument("--train-frac", type=float, default=0.8) # 训练集比例
    parser.add_argument("--val-frac", type=float, default=0.1) # 验证集比例
    return parser.parse_args()
"""python train_tg.py --csv data/PSMILES_Tg_only.csv --polybert-dir ./polybert --epochs 30 --polybert-train-last-n 2 --lambda-tg 0.5 --output checkpoints/modelv3_tg.pt"""

from typing import Optional


def prepare_device(preferred: Optional[str]):
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model, base_lr, polybert_lr, weight_decay):
    polybert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("polybert."):
            polybert_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": base_lr, "weight_decay": weight_decay}]
    if polybert_params:
        groups.append({"params": polybert_params, "lr": polybert_lr, "weight_decay": weight_decay})
    return AdamW(groups)


def run_epoch(model, loader, optimizer, pad_id, device, kl_weight, lambda_tg):
    model.train()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tg = batch["tg"].to(device).unsqueeze(-1)

        logits, mu, logvar, tg_pred = model(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            conditions=tg,
            encoder_attention_mask=attention_mask,
        )
        loss_rec = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        loss_kld = kld_loss(mu, logvar)
        loss_tg = torch.tensor(0.0, device=device)
        if tg_pred is not None:
            loss_tg = torch.nn.functional.mse_loss(tg_pred.squeeze(-1), tg.squeeze(-1))
        loss = loss_rec + kl_weight * loss_kld + lambda_tg * loss_tg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * input_ids.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, pad_id, device, kl_weight, lambda_tg):
    model.eval()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tg = batch["tg"].to(device).unsqueeze(-1)

        logits, mu, logvar, tg_pred = model(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            conditions=tg,
            encoder_attention_mask=attention_mask,
        )
        loss_rec = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        loss_kld = kld_loss(mu, logvar)
        loss_tg = torch.tensor(0.0, device=device)
        if tg_pred is not None:
            loss_tg = torch.nn.functional.mse_loss(tg_pred.squeeze(-1), tg.squeeze(-1))
        total += (loss_rec + kl_weight * loss_kld + lambda_tg * loss_tg).item() * input_ids.size(0)
    return total / len(loader.dataset)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = prepare_device(args.device)

    tokenizer = PolyBertTokenizer(str(args.polybert_dir))
    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = split_dataframe(df, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)

    train_loader, tg_stats = make_loader_with_tg(
        train_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
    )
    val_loader, _ = make_loader_with_tg(
        val_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        tg_stats=tg_stats,
    )
    test_loader, _ = make_loader_with_tg(
        test_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        tg_stats=tg_stats,
    )

    polybert = AutoModel.from_pretrained(str(args.polybert_dir)).to(device)
    configure_polybert_finetuning(
        polybert,
        train_last_n_layers=args.polybert_train_last_n,
    )

    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        emb_dim=args.emb_dim,
        encoder_hid_dim=polybert.config.hidden_size,
        decoder_hid_dim=args.decoder_hid_dim,
        z_dim=args.z_dim,
        cond_dim=1,
        cond_latent_dim=args.cond_latent_dim,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        drop=args.dropout,
        use_polybert=True,
        polybert=polybert,
        freeze_polybert=False,
        polybert_pooling="cls",
        max_len=args.max_len,
        use_tg_regression=True,
        tg_hidden_dim=args.tg_hidden_dim,
    )
    model = ConditionalVAESmiles(**model_kwargs).to(device)

    optimizer = build_optimizer(model, args.lr, args.polybert_lr, args.weight_decay)

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        kl_w = min(1.0, epoch / max(1, args.kl_warmup))
        train_loss = run_epoch(model, train_loader, optimizer, tokenizer.pad_id, device, kl_w, args.lambda_tg)
        val_loss = evaluate(model, val_loader, tokenizer.pad_id, device, kl_w, args.lambda_tg)
        print(f"[{epoch}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f} KLw={kl_w:.2f}")

        if val_loss + 1e-3 < best_val:
            best_val = val_loss
            payload = {
                "model": model.state_dict(),
                "tokenizer": tokenizer.get_vocab(),
                "tokenizer_settings": {
                    "pad_id": tokenizer.pad_id,
                    "bos_id": tokenizer.bos_id,
                    "eos_id": tokenizer.eos_id,
                },
                "tg_stats": {"mean": tg_stats.mean, "std": tg_stats.std},
                "model_kwargs": {
                    k: v
                    for k, v in model_kwargs.items()
                    if k not in {"polybert"}
                },
                "config": vars(args),
                "best_val_loss": best_val,
            }
            torch.save(payload, args.output)
            print(f"  Saved checkpoint to {args.output}")

    test_loss = evaluate(model, test_loader, tokenizer.pad_id, device, kl_weight=1.0, lambda_tg=args.lambda_tg)
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
