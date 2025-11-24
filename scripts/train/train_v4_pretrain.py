"""
预训练阶段：使用无 Tg 标签的大规模 pSMILES 数据（如 PI1M_v2）做重构 + KL 训练。
条件向量用全零占位，这样模型结构与后续带 Tg 微调保持一致，便于加载权重。
"""

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent # 自动定位到项目根目录
sys.path.append(str(ROOT / "src"))  # 允许从 src 目录直接导入模块

from dataset import make_loader
from tokenizer import PolyBertTokenizer
from train import (
    kld_loss,
    set_seed,
    split_dataframe,
    configure_polybert_finetuning,
)


def resolve_model_class(model_size: str):
    if model_size == "base":
        from modelv4 import ConditionalVAESmiles
    elif model_size == "medium":
        from modelv4_medium import ConditionalVAESmiles
    elif model_size == "premium":
        from modelv4_premium import ConditionalVAESmiles
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    return ConditionalVAESmiles


def build_param_groups(model, base_lr: float, polybert_lr: Optional[float], weight_decay: float):
    """为 polyBERT 和其他模块设置不同学习率。"""
    polybert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("polybert."):
            polybert_params.append(param)
        else:
            other_params.append(param)

    groups = [{"params": other_params, "lr": base_lr, "weight_decay": weight_decay}]
    if polybert_params and polybert_lr is not None:
        groups.append({"params": polybert_params, "lr": polybert_lr, "weight_decay": weight_decay})
    return groups


def train_one_epoch(model, loader: DataLoader, opt, kl_weight, pad_id, device, *, scaler=None, scheduler=None):
    model.train()
    total = 0.0
    device_type = device.type if isinstance(device, torch.device) else device
    use_amp = scaler is not None and device_type.startswith("cuda")
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)  # [B, T]
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        conditions = torch.zeros(input_ids.size(0), 1, device=device)  # 预训练阶段条件向量占位

        ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            logits, mu, logvar, _ = model(
                encoder_input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                conditions=conditions,
                encoder_attention_mask=attention_mask,
            )
            loss_rec = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id,
            )
            loss_kld = kld_loss(mu, logvar)
            loss = loss_rec + kl_weight * loss_kld

        opt.zero_grad(set_to_none=True)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * input_ids.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def val_loss(model, loader: DataLoader, kl_weight, pad_id, device):
    model.eval()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        conditions = torch.zeros(input_ids.size(0), 1, device=device)

        logits, mu, logvar, _ = model(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            conditions=conditions,
            encoder_attention_mask=attention_mask,
        )
        loss_rec = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id,
        )
        loss_kld = kld_loss(mu, logvar)
        total += (loss_rec + kl_weight * loss_kld).item() * input_ids.size(0)
    return total / len(loader.dataset)


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Pretrain ConditionalVAESmiles on unlabeled pSMILES data.")
    parser.add_argument("--csv", type=Path, default=Path("data/PI1M_v2_psmiles.csv"), help="输入 CSV 路径（需包含 PSMILES 列）")
    parser.add_argument("--col", type=str, default="PSMILES", help="pSMILES 列名")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4, help="非 polyBERT 参数的学习率")
    parser.add_argument("--polybert-lr", type=float, default=1e-5, help="polyBERT 参数学习率")
    parser.add_argument("--polybert-train-last-n", type=int, default=2, help="解冻 polyBERT 最后 N 层；0 表示全冻")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/pretrain_modelv4.pt"))
    parser.add_argument("--polybert-dir", type=str, default="./polybert", help="polyBERT 权重路径或 HF 名称")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "medium", "premium"], help="modelv4 容量等级")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None):
    args = parse_args(argv)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = split_dataframe(df, train_frac=0.9, val_frac=0.05, seed=42)

    tokenizer = PolyBertTokenizer(args.polybert_dir if Path(args.polybert_dir).exists() else "kuelumbus/polyBERT")
    train_loader = make_loader(
        train_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        col=args.col,
        max_len=args.max_len,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = make_loader(
        val_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        col=args.col,
        max_len=args.max_len,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = make_loader(
        test_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        col=args.col,
        max_len=args.max_len,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    polybert_encoder = AutoModel.from_pretrained(args.polybert_dir)
    trainable_polybert_params = configure_polybert_finetuning(
        polybert_encoder,
        train_last_n_layers=args.polybert_train_last_n,
    )
    if trainable_polybert_params:
        print(f"Fine-tuning last {args.polybert_train_last_n} polyBERT layers (~{sum(p.numel() for p in trainable_polybert_params):,} params).")
    else:
        print("polyBERT kept frozen.")

    bos_token_id = tokenizer.bos_id
    eos_token_id = tokenizer.eos_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must provide BOS/CLS and EOS/SEP token ids.")

    ModelCls = resolve_model_class(args.model_size)
    model = ModelCls(
        vocab_size=tokenizer.vocab_size,
        cond_dim=1,
        pad_id=tokenizer.pad_id,
        bos_id=bos_token_id,
        eos_id=eos_token_id,
        drop=args.dropout,
        use_polybert=True,
        polybert=polybert_encoder,
        freeze_polybert=False,
        polybert_pooling="cls",
        use_tg_regression=False,  # 预训练无需 Tg head
        max_len=args.max_len,
    ).to(device)

    opt = torch.optim.AdamW(build_param_groups(model, args.lr, args.polybert_lr, weight_decay=0.01))
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best = float("inf")
    patience, bad = 3, 0
    for ep in range(1, args.epochs + 1):
        kl_w = min(1.0, ep / 10.0)  # 线性 KL 退火
        tr = train_one_epoch(model, train_loader, opt, kl_w, tokenizer.pad_id, device, scaler=scaler, scheduler=scheduler)
        va = val_loss(model, val_loader, kl_w, tokenizer.pad_id, device)
        print(f"[{ep}/{args.epochs}] train {tr:.4f}  val {va:.4f}  kl_w={kl_w:.2f}")

        if va < best - 1e-3:
            best, bad = va, 0
            args.output.parent.mkdir(exist_ok=True)
            model_kwargs = {
                "vocab_size": tokenizer.vocab_size,
                "emb_dim": model.emb_dim,
                "encoder_hid_dim": model.encoder_hidden_dim,
                "decoder_hid_dim": model.decoder_hid_dim,
                "z_dim": model.z_dim,
                "cond_dim": 1,
                "cond_latent_dim": model.cond_latent_dim,
                "pad_id": tokenizer.pad_id,
                "bos_id": bos_token_id,
                "eos_id": eos_token_id,
                "drop": args.dropout,
                "use_polybert": True,
                "freeze_polybert": False,
                "polybert_pooling": "cls",
                "use_tg_regression": False,
                "max_len": args.max_len,
                "num_decoder_layers": model.num_decoder_layers,
                "decoder_nhead": model.decoder_nhead,
                "decoder_ff_mult": model.decoder_ff_mult,
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.get_vocab(),
                    "pad_token_id": tokenizer.pad_id,
                    "bos_token_id": tokenizer.bos_id,
                    "eos_token_id": tokenizer.eos_id,
                    "tokenizer_name": args.polybert_dir,
                    "use_polybert": True,
                    "model_kwargs": model_kwargs,
                    "model_size": args.model_size,
                },
                args.output,
            )
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    test_loss_val = val_loss(model, test_loader, kl_weight=1.0, pad_id=tokenizer.pad_id, device=device)
    print(f"Test loss {test_loss_val:.4f}")


if __name__ == "__main__":
    main()
