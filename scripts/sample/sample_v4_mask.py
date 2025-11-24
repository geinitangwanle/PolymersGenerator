import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel

repo = Path(__file__).resolve().parent
sys.path.append(str(repo / "src"))

from src.modelv4 import ConditionalVAESmiles  # noqa: E402
from src.dataset_tg import TgStats, compute_tg_stats  # noqa: E402
from src.syntax_mask import SyntaxMasker  # noqa: E402
from src.tokenizer import PolyBertTokenizer  # noqa: E402
from src.train import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Sample SMILES with grammar mask using finetuned modelv4 (ConditionalVAESmiles).")
    parser.add_argument("--checkpoint", type=Path, default=repo / "checkpoints/finetune_tg_modelv4.pt",
                        help="Path to the finetuned modelv4 checkpoint.")
    parser.add_argument("--polybert-dir", type=Path, default=repo / "polybert",
                        help="Directory containing the polyBERT weights/tokenizer.")
    parser.add_argument("--data-csv", type=Path, default=repo / "data/PSMILES_Tg_only.csv",
                        help="CSV file used during training (for novelty metric and Tg stats fallback).")
    parser.add_argument("--data-col", type=str, default="PSMILES",
                        help="Column name in the CSV that holds SMILES strings.")
    parser.add_argument("--col-tg", type=str, default="Tg",
                        help="Column name for Tg values (used for normalization if stats not in checkpoint).")
    parser.add_argument("--target-tg", type=float, nargs="+", required=True,
                        help="Target Tg values (Kelvin) for conditional sampling.")
    parser.add_argument("--num-per-target", type=int, default=128,
                        help="Number of samples to generate per Tg target.")
    parser.add_argument("--max-len", type=int, default=256,
                        help="Maximum decoding length when sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling parameter.")
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs_masked",
                        help="Directory to store sampled SMILES and metrics.")
    parser.add_argument("--samples-file", type=str, default="sampled_smiles_masked.csv",
                        help="Filename for the sampled SMILES CSV.")
    parser.add_argument("--metrics-file", type=str, default="sample_metrics_masked.json",
                        help="Filename for the metrics JSON.")
    parser.add_argument("--disable-grammar-mask", action="store_true",
                        help="Turn off grammar mask (for ablation).")
    return parser.parse_args()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args, device) -> Tuple[ConditionalVAESmiles, PolyBertTokenizer, Optional[TgStats]]:
    set_seed(args.seed)
    # 显式设置 weights_only=False，避免未来默认变更带来的行为差异
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    polybert_name = ckpt.get("tokenizer_name", str(args.polybert_dir))
    tokenizer = PolyBertTokenizer(polybert_name)
    polybert = AutoModel.from_pretrained(polybert_name).to(device)

    if "model_kwargs" in ckpt:
        model_kwargs = ckpt["model_kwargs"]
        model_kwargs.update({
            "vocab_size": tokenizer.vocab_size,
            "polybert": polybert,
            "use_polybert": True,
            "pad_id": tokenizer.pad_id,
            "bos_id": tokenizer.bos_id,
            "eos_id": tokenizer.eos_id,
        })
        model = ConditionalVAESmiles(**model_kwargs).to(device)
    else:
        model = ConditionalVAESmiles(
            vocab_size=tokenizer.vocab_size,
            emb_dim=256,
            encoder_hid_dim=polybert.config.hidden_size,
            decoder_hid_dim=512,
            z_dim=128,
            cond_dim=1,
            cond_latent_dim=32,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            drop=0.1,
            use_polybert=True,
            polybert=polybert,
            freeze_polybert=True,
            polybert_pooling="cls",
            use_tg_regression=False,
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tg_stats = None
    if "tg_stats" in ckpt:
        tg_stats = TgStats(**ckpt["tg_stats"])
    return model, tokenizer, tg_stats


def to_rdkit(smiles: str):
    # 替换聚合物占位符使 RDKit 能解析
    return Chem.MolFromSmiles(smiles.replace("[*]", "[Xe]"))


def _apply_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, probs.size(-1))
    topk_vals, topk_idx = probs.topk(k, dim=-1)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(1, topk_idx, topk_vals)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return filtered


def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(1, sorted_idx, sorted_probs)
    return filtered


@torch.no_grad()
def sample_smiles_with_mask(
    model: ConditionalVAESmiles,
    tokenizer: PolyBertTokenizer,
    syntax_masker: Optional[SyntaxMasker],
    tg_stats: TgStats,
    targets: List[float],
    device: torch.device,
    num_per_target: int,
    max_len: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> List[str]:
    cond_dim = model.cond_encoder[0].in_features
    all_conditions = []
    for t in targets:
        norm = tg_stats.normalize([t])[0]
        vec = torch.full((num_per_target, cond_dim), float(norm), device=device)
        all_conditions.append(vec)
    conditions = torch.cat(all_conditions, dim=0)  # [B, cond_dim]

    num_samples = conditions.size(0)
    z_base = torch.randn(num_samples, model.z_dim, device=device)
    z_concat, cond_latent = model._prepare_latent(z_base, conditions)
    gamma, beta = model._compute_film(cond_latent)
    memory = model.latent_proj(z_concat)

    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must provide both BOS and EOS/SEP token ids for sampling.")

    cur = torch.full((num_samples, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(num_samples, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        T = cur.size(1)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        tgt = model.emb(cur) + model.pos_emb(pos_ids)
        tgt_mask = model._causal_mask(T, device=device)

        y = model._run_decoder(
            tgt=tgt,
            memory=memory.unsqueeze(1).expand(num_samples, T, -1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(cur == tokenizer.pad_id),
            gamma=gamma,
            beta=beta,
        )
        logits = model.out(y[:, -1:])
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits.squeeze(1), dim=-1)

        if syntax_masker is not None:
            masks = syntax_masker.batch_mask(cur, finished=finished, device=device)
            mask_f = masks.float()
            probs = probs * mask_f
            row_sums = probs.sum(dim=-1, keepdim=True)
            fallback = torch.where(
                mask_f.sum(dim=-1, keepdim=True) > 0,
                mask_f / mask_f.sum(dim=-1, keepdim=True).clamp(min=1e-8),
                torch.full_like(mask_f, 1.0 / mask_f.size(1)),
            )
            probs = torch.where(row_sums > 0, probs, fallback)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if top_k is not None:
            probs = _apply_top_k(probs, top_k)
        elif top_p is not None:
            probs = _apply_top_p(probs, top_p)

        next_tok = torch.multinomial(probs, num_samples=1)
        next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, eos_id), next_tok)
        finished |= next_tok.squeeze(1) == eos_id
        cur = torch.cat([cur, next_tok], dim=1)
        if finished.all():
            break

    decoded = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in cur.cpu()]
    return decoded


def compute_metrics(samples: List[str], train_set: set) -> Dict[str, float]:
    unique = set(samples)
    valid = [s for s in samples if to_rdkit(s)]
    novelty_set = [s for s in unique if s not in train_set]
    return {
        "num_samples": len(samples),
        "num_valid": len(valid),
        "num_unique": len(unique),
        "num_novel": len(novelty_set),
        "validity": len(valid) / len(samples) if samples else 0.0,
        "uniqueness": len(unique) / len(samples) if samples else 0.0,
        "novelty": len(novelty_set) / len(unique) if unique else 0.0,
    }


def save_samples(samples: List[str], train_set: set, output_path: Path):
    seen = set()
    records = []
    for idx, s in enumerate(samples):
        mol = to_rdkit(s)
        is_valid = mol is not None
        is_unique = s not in seen
        in_training = s in train_set
        records.append(
            {
                "sample_id": idx,
                "smiles": s,
                "is_valid": is_valid,
                "is_unique": is_unique,
                "in_training_set": in_training,
            }
        )
        seen.add(s)

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def save_metrics(metrics: Dict[str, float], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    device = prepare_device()
    model, tokenizer, tg_stats = load_model(args, device)
    syntax_masker = None if args.disable_grammar_mask else SyntaxMasker(
        tokenizer, vocab_size=model.out.out_features
    )

    df = pd.read_csv(args.data_csv)
    if args.data_col not in df.columns:
        raise ValueError(f"Column '{args.data_col}' not found in {args.data_csv}")
    if tg_stats is None:
        if args.col_tg not in df.columns:
            raise ValueError(f"Column '{args.col_tg}' not found in {args.data_csv} to compute Tg stats.")
        tg_stats = compute_tg_stats(df[args.col_tg].astype(float))

    train_set = set(df[args.data_col].astype(str))

    samples = sample_smiles_with_mask(
        model=model,
        tokenizer=tokenizer,
        syntax_masker=syntax_masker,
        tg_stats=tg_stats,
        targets=args.target_tg,
        device=device,
        num_per_target=args.num_per_target,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    output_dir = args.output_dir
    samples_path = output_dir / args.samples_file
    metrics_path = output_dir / args.metrics_file

    save_samples(samples, train_set, samples_path)
    metrics = compute_metrics(samples, train_set)
    save_metrics(metrics, metrics_path)

    print(f"Saved {len(samples)} samples to {samples_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(f"Metrics file written to {metrics_path}")


if __name__ == "__main__":
    main()
