import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from transformers import AutoModel

repo = Path("/share/home/u23514/HTY/PolymersGenerator")
sys.path.append(str(repo / "src"))

from src.dataset_tg import make_loader_with_tg
from src.modelv3 import ConditionalVAESmiles
from src.tokenizer import PolyBertTokenizer
from src.train import (
    configure_polybert_finetuning,
    kld_loss,
    set_seed,
    split_dataframe,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training for Tg-conditional VAE on HPC")
    parser.add_argument("--csv", type=Path, default=repo / "data/PSMILES_Tg_only.csv")
    parser.add_argument("--col-smiles", type=str, default="PSMILES")
    parser.add_argument("--col-tg", type=str, default="Tg")
    parser.add_argument("--polybert-dir", type=Path, default=repo / "polybert")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--polybert-lr", type=float, default=1e-5)
    parser.add_argument("--polybert-train-last-n", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-warmup", type=int, default=10)
    parser.add_argument("--lambda-tg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=repo / "checkpoints/modelv3_tg.pt")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--decoder-hid-dim", type=int, default=512)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--cond-latent-dim", type=int, default=32)
    parser.add_argument("--tg-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    return parser.parse_args()


def build_param_groups(model, base_lr, polybert_lr, weight_decay):
    polybert_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.split("module.", 1)[-1]
        if clean_name.startswith("polybert."):
            polybert_params.append(param)
        else:
            other_params.append(param)
    groups = [{"params": other_params, "lr": base_lr, "weight_decay": weight_decay}]
    if polybert_params:
        groups.append({"params": polybert_params, "lr": polybert_lr, "weight_decay": weight_decay})
    return groups


def run_epoch(model, loader, optimizer, pad_id, device, kl_weight, lambda_tg, scaler=None):
    model.train()
    total = 0.0
    device_type = device.type if isinstance(device, torch.device) else device
    use_amp = scaler is not None and device_type.startswith("cuda")

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tg = batch["tg"].to(device, non_blocking=True).unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=use_amp):
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

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        tg = batch["tg"].to(device, non_blocking=True).unsqueeze(-1)

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


def main(rank, world_size, args):
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] device: {device}")

    csv_path = args.csv
    tokenizer = PolyBertTokenizer(str(args.polybert_dir))

    df = pd.read_csv(csv_path)
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
        distributed=(world_size > 1),
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
        distributed=False,
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
        distributed=False,
    )

    polybert = AutoModel.from_pretrained(str(args.polybert_dir)).to(device)
    configure_polybert_finetuning(polybert, train_last_n_layers=args.polybert_train_last_n)

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

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optimizer = AdamW(build_param_groups(model, args.lr, args.polybert_lr, args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        kl_weight = min(1.0, epoch / max(1, args.kl_warmup))
        train_loss = run_epoch(model, train_loader, optimizer, tokenizer.pad_id, device, kl_weight, args.lambda_tg, scaler)
        val_loss = evaluate(model, val_loader, tokenizer.pad_id, device, kl_weight, args.lambda_tg)
        if rank == 0:
            print(f"[Epoch {epoch}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f} KL={kl_weight:.2f}")

        if rank == 0 and val_loss + 1e-3 < best_val:
            best_val = val_loss
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            payload = {
                "model": state_dict,
                "tokenizer": tokenizer.get_vocab(),
                "tokenizer_settings": {
                    "pad_id": tokenizer.pad_id,
                    "bos_id": tokenizer.bos_id,
                    "eos_id": tokenizer.eos_id,
                },
                "tg_stats": {"mean": tg_stats.mean, "std": tg_stats.std},
                "model_kwargs": {k: v for k, v in model_kwargs.items() if k != "polybert"},
                "config": vars(args),
                "best_val_loss": best_val,
            }
            torch.save(payload, args.output)
            print(f"[Rank 0] Saved checkpoint to {args.output}")

    if rank == 0:
        test_loss = evaluate(model, test_loader, tokenizer.pad_id, device, kl_weight=1.0, lambda_tg=args.lambda_tg)
        print(f"[Rank 0] Test loss: {test_loss:.4f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.spawn(main, args=(n_gpus, args), nprocs=n_gpus, join=True)
    else:
        main(rank=0, world_size=1, args=args)
