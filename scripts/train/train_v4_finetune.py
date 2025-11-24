"""
带 Tg 微调脚本：在预训练的 ConditionalVAESmiles 上加载无标签权重，再用带 Tg 的数据集联合优化重构/KL/Tg 回归。
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_ROOT.parent.parent  # .../PolymersGenerator
sys.path.append(str(PROJ_ROOT / "src"))

from tokenizer import PolyBertTokenizer  # noqa: E402
from dataset_tg import make_loader_with_tg, TgStats  # noqa: E402
from train import (  # noqa: E402
    kld_loss,
    set_seed,
    split_dataframe,
    configure_polybert_finetuning,
)


def build_param_groups(model, base_lr: float, polybert_lr: Optional[float], weight_decay: float):
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


def train_one_epoch(model, loader, opt, kl_weight, lambda_tg, pad_id, device, *, scaler=None, scheduler=None):
    model.train()
    total = 0.0
    device_type = device.type if isinstance(device, torch.device) else device
    use_amp = scaler is not None and device_type.startswith("cuda")
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        conditions = batch["tg"].unsqueeze(-1).to(device)

        ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            logits, mu, logvar, tg_pred = model(
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
            loss_tg = torch.nn.functional.mse_loss(tg_pred.squeeze(-1), conditions.squeeze(-1)) if tg_pred is not None else 0.0
            loss = loss_rec + kl_weight * loss_kld + lambda_tg * loss_tg

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
def val_loss(model, loader, kl_weight, lambda_tg, pad_id, device):
    model.eval()
    total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        conditions = batch["tg"].unsqueeze(-1).to(device)

        logits, mu, logvar, tg_pred = model(
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
        loss_tg = torch.nn.functional.mse_loss(tg_pred.squeeze(-1), conditions.squeeze(-1)) if tg_pred is not None else 0.0
        total += (loss_rec + kl_weight * loss_kld + lambda_tg * loss_tg).item() * input_ids.size(0)
    return total / len(loader.dataset)


def parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Finetune ConditionalVAESmiles with Tg labels starting from pretrained weights.")
    parser.add_argument("--csv", type=Path, default=Path("data/PSMILES_Tg_only.csv"), help="带 Tg 的数据集 CSV 路径")
    parser.add_argument("--col-smiles", type=str, default="PSMILES", help="SMILES 列名")
    parser.add_argument("--col-tg", type=str, default="Tg", help="Tg 列名")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4, help="非 polyBERT 参数的学习率")
    parser.add_argument("--polybert-lr", type=float, default=5e-6, help="polyBERT 参数学习率")
    parser.add_argument("--polybert-train-last-n", type=int, default=2, help="解冻 polyBERT 最后 N 层；0 表示全冻")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-tg", type=float, default=0.5, help="Tg 回归损失系数")
    parser.add_argument("--pretrained", type=Path, default=Path("checkpoints/pretrain_modelv4.pt"), help="预训练 checkpoint 路径")
    parser.add_argument("--polybert-dir", type=str, default="./polybert", help="polyBERT 权重路径或 HF 名称")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/finetune_tg_modelv4.pt"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model-size", type=str, default="base", choices=["base", "medium", "premium"], help="modelv4 容量等级")
    return parser.parse_args(argv)


def load_pretrained(model, ckpt_path, device):
    if not ckpt_path.exists():
        print(f"Pretrained checkpoint not found: {ckpt_path}, training from scratch.")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"Missing keys (expected for Tg head if absent): {missing}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")


def main(argv: Optional[Iterable[str]] = None):
    args = parse_args(argv)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    train_df, val_df, test_df = split_dataframe(df, train_frac=0.8, val_frac=0.1, seed=42)

    tokenizer = PolyBertTokenizer(args.polybert_dir if Path(args.polybert_dir).exists() else "kuelumbus/polyBERT")
    train_loader, tg_stats = make_loader_with_tg(
        train_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        num_workers=args.num_workers,
        distributed=False,
    )
    val_loader, _ = make_loader_with_tg(
        val_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        num_workers=args.num_workers,
        distributed=False,
        tg_stats=tg_stats,
    )
    test_loader, _ = make_loader_with_tg(
        test_df,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        num_workers=args.num_workers,
        distributed=False,
        tg_stats=tg_stats,
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

    if args.model_size == "base":
        from modelv4 import ConditionalVAESmiles as ModelCls
    elif args.model_size == "medium":
        from modelv4_medium import ConditionalVAESmiles as ModelCls
    elif args.model_size == "premium":
        from modelv4_premium import ConditionalVAESmiles as ModelCls
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

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
        use_tg_regression=True,
        max_len=args.max_len,
    ).to(device)

    load_pretrained(model, args.pretrained, device)

    opt = torch.optim.AdamW(build_param_groups(model, args.lr, args.polybert_lr, weight_decay=0.01))
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best = float("inf")
    patience, bad = 3, 0
    for ep in range(1, args.epochs + 1):
        kl_w = min(1.0, ep / 10.0)
        tr = train_one_epoch(model, train_loader, opt, kl_w, args.lambda_tg, tokenizer.pad_id, device, scaler=scaler, scheduler=scheduler)
        va = val_loss(model, val_loader, kl_w, args.lambda_tg, tokenizer.pad_id, device)
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
                "use_tg_regression": True,
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
                    "tg_stats": {"mean": tg_stats.mean, "std": tg_stats.std},
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

    test_loss_val = val_loss(model, test_loader, kl_weight=1.0, lambda_tg=args.lambda_tg, pad_id=tokenizer.pad_id, device=device)
    print(f"Test loss {test_loss_val:.4f}")


if __name__ == "__main__":
    main()
