#!/usr/bin/env python
"""Compute & visualize latent space (PCA scatter + histograms) for modelv4 base/medium/premium."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJ_ROOT))  # allow `src` imports when run as a script

from src.dataset_tg import TgStats, make_loader_with_tg  # noqa: E402
from src.tokenizer import PolyBertTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键计算并可视化 modelv4 潜空间（PCA + 直方图）。")
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["base", "medium", "premium"],
        help="模型容量版本：base/medium/premium。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="模型 checkpoint 路径；若提供 --latent 且无需重算，可省略。",
    )
    parser.add_argument(
        "--latent",
        type=Path,
        default=None,
        help="已导出的 latent .npz（包含 mu/logvar/tg），若未提供则会先计算。",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJ_ROOT / "data/PSMILES_Tg_only.csv",
        help="数据 CSV（含 PSMILES/Tg），仅在需要重新计算时使用。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJ_ROOT / "sample_output",
        help="保存可视化图/latent 的目录。",
    )
    parser.add_argument("--pca-sample", type=int, default=10000, help="PCA 子采样数量上限。")
    parser.add_argument("--force-recompute", action="store_true", help="即便有 --latent 也重新计算。")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批大小（计算 latent 时）。")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数。")
    parser.add_argument("--max-len", type=int, default=256, help="SMILES 最大长度。")
    parser.add_argument("--col-smiles", type=str, default="PSMILES", help="SMILES 列名。")
    parser.add_argument("--col-tg", type=str, default="Tg", help="Tg 列名。")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 计算。")
    return parser.parse_args()


def select_model_cls(model_size: Literal["base", "medium", "premium"]):
    if model_size == "base":
        from src.modelv4 import ConditionalVAESmiles
    elif model_size == "medium":
        from src.modelv4_medium import ConditionalVAESmiles
    else:
        from src.modelv4_premium import ConditionalVAESmiles
    return ConditionalVAESmiles


def load_model_and_tokenizer(
    model_size: Literal["base", "medium", "premium"], checkpoint: Path, device: torch.device
) -> Tuple[torch.nn.Module, PolyBertTokenizer, Optional[TgStats]]:
    # Explicit weights_only=False to keep checkpoint extras (avoids FutureWarning)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    tokenizer = PolyBertTokenizer(str(PROJ_ROOT / "polybert"))
    polybert = AutoModel.from_pretrained(str(PROJ_ROOT / "polybert")).to(device)

    model_kwargs = ckpt["model_kwargs"].copy()
    model_kwargs.update(
        {
            "vocab_size": tokenizer.vocab_size,
            "polybert": polybert,
            "use_polybert": True,
        }
    )

    ModelCls = select_model_cls(model_size)
    model = ModelCls(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tg_stats = TgStats(**ckpt["tg_stats"]) if "tg_stats" in ckpt else None
    return model, tokenizer, tg_stats


def export_latent(args: argparse.Namespace) -> Path:
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] using device: {device}")

    model, tokenizer, tg_stats = load_model_and_tokenizer(args.model_size, args.checkpoint, device)
    loader, computed_stats = make_loader_with_tg(
        args.data,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        max_len=args.max_len,
        tg_stats=tg_stats,  # if None, stats will be computed from data
    )

    all_mu, all_logvar, all_tg = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx % 10 == 0:
                print(f"[info] batch {idx+1}/{len(loader)}")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mu, logvar = model.encode(input_ids, attention_mask)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            stats_to_use = tg_stats or computed_stats
            all_tg.append(stats_to_use.denormalize(batch["tg"].cpu().numpy()))

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    tg = np.concatenate(all_tg, axis=0)

    out_path = args.output_dir / f"latent_{args.model_size}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mu=mu, logvar=logvar, tg=tg)
    print(f"[done] saved mu/logvar/tg to: {out_path} | shape={mu.shape}")
    return out_path


def load_latent(args: argparse.Namespace) -> Path:
    if args.latent is not None and args.latent.exists() and not args.force_recompute:
        print(f"[info] load latent from: {args.latent}")
        return args.latent
    if args.checkpoint is None:
        raise ValueError("需要 checkpoint 才能计算 latent（或指定已存在的 --latent）。")

    print("[info] computing latent ...")
    return export_latent(args)


def subsample(arr: np.ndarray, max_n: int) -> np.ndarray:
    if len(arr) <= max_n:
        return arr
    idx = np.random.RandomState(42).choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def plot(args: argparse.Namespace, latent_path: Path):
    data = np.load(latent_path)
    mu, logvar, tg = data["mu"], data["logvar"], data["tg"]
    print(f"[info] loaded mu/logvar/tg shapes: {mu.shape}, {logvar.shape}, {tg.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # PCA scatter
    mu_for_pca = subsample(mu, args.pca_sample)
    tg_for_pca = subsample(tg, args.pca_sample)
    mu_2d = PCA(n_components=2).fit_transform(mu_for_pca)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=tg_for_pca, cmap="inferno", s=8, alpha=0.7)
    plt.colorbar(sc, label="Tg (K)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Latent μ PCA ({args.model_size})")
    plt.tight_layout()
    pca_path = args.output_dir / f"latent_{args.model_size}_pca.png"
    plt.savefig(pca_path, dpi=300)
    plt.close()
    print(f"[done] saved PCA scatter: {pca_path}")

    # Distribution of mu
    plt.figure(figsize=(7, 4))
    plt.hist(mu.reshape(-1), bins=100, alpha=0.75)
    plt.xlabel("μ values")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of latent means ({args.model_size})")
    plt.tight_layout()
    mu_path = args.output_dir / f"latent_{args.model_size}_mu_hist.png"
    plt.savefig(mu_path, dpi=300)
    plt.close()
    print(f"[done] saved mu histogram: {mu_path}")

    # Posterior variance distribution
    plt.figure(figsize=(7, 4))
    plt.hist(np.exp(logvar).reshape(-1), bins=100, alpha=0.75)
    plt.xlabel("Posterior variance exp(logvar)")
    plt.ylabel("Frequency")
    plt.title(f"Posterior variance distribution ({args.model_size})")
    plt.tight_layout()
    var_path = args.output_dir / f"latent_{args.model_size}_var_hist.png"
    plt.savefig(var_path, dpi=300)
    plt.close()
    print(f"[done] saved variance histogram: {var_path}")


def main():
    args = parse_args()
    latent_path = load_latent(args)
    plot(args, latent_path)


if __name__ == "__main__":
    main()
