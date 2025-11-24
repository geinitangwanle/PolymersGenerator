"""
无条件采样脚本：用于加载预训练的 ConditionalVAESmiles（无 Tg 头）并生成 pSMILES。
预训练权重来自 `train_pretrain.py`，条件向量统一置零。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_ROOT.parent.parent  # .../PolymersGenerator
sys.path.append(str(PROJ_ROOT / "src"))

from tokenizer import PolyBertTokenizer  # noqa: E402
from train import set_seed  # noqa: E402
from modelv4 import ConditionalVAESmiles  # noqa: E402


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


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional sampling from a pretrained ConditionalVAESmiles.")
    parser.add_argument("--checkpoint", type=Path, default=PROJ_ROOT / "checkpoints/pretrain_modelv4.pt",
                        help="Path to the pretrained checkpoint from train_pretrain.py")
    parser.add_argument("--polybert-dir", type=Path, default=PROJ_ROOT / "polybert",
                        help="Directory containing the polyBERT weights/tokenizer (match pretraining).")
    parser.add_argument("--model-size", type=str, default=None, choices=["base", "medium", "premium"],
                        help="Optional override for model capacity; if None, infer from checkpoint.")
    parser.add_argument("--data-csv", type=Path, default=None,
                        help="Optional CSV to compute novelty metrics (column specified by --data-col).")
    parser.add_argument("--data-col", type=str, default="PSMILES",
                        help="Column name with SMILES in data-csv.")
    parser.add_argument("--num-samples", type=int, default=512,
                        help="Number of SMILES to generate.")
    parser.add_argument("--max-len", type=int, default=256,
                        help="Maximum decoding length.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (applied before top-p).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=PROJ_ROOT / "outputs_pretrain",
                        help="Directory to store outputs.")
    parser.add_argument("--samples-file", type=str, default="samples_pretrain.csv",
                        help="Filename for sampled SMILES CSV.")
    parser.add_argument("--metrics-file", type=str, default="metrics_pretrain.json",
                        help="Filename for metrics JSON.")
    return parser.parse_args()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args, device) -> Tuple[torch.nn.Module, PolyBertTokenizer]:
    set_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # 决定 polyBERT 源：优先 checkpoint 里的名字，否则用传入路径
    polybert_name = ckpt.get("tokenizer_name", str(args.polybert_dir))
    tokenizer = PolyBertTokenizer(polybert_name)
    polybert = AutoModel.from_pretrained(polybert_name).to(device)

    ckpt_model_size = ckpt.get("model_size")
    model_size = args.model_size or ckpt_model_size or "base"
    ModelCls = resolve_model_class(model_size)

    if "model_kwargs" in ckpt:
        model_kwargs = ckpt["model_kwargs"].copy()
        # 以 checkpoint 超参为主，避免手工 override 出现不一致
        model_kwargs.update(
            {
                "vocab_size": tokenizer.vocab_size,
                "polybert": polybert,
                "use_polybert": True,
                "pad_id": tokenizer.pad_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            }
        )
        model = ModelCls(**model_kwargs).to(device)
    else:
        # 回落到旧的默认配置（base）
        model = ModelCls(
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
            max_len=args.max_len,
        ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def to_rdkit(smiles: str):
    return Chem.MolFromSmiles(smiles.replace("[*]", "[Xe]"))


@torch.no_grad()
def sample_smiles(
    model: torch.nn.Module,
    tokenizer: PolyBertTokenizer,
    device: torch.device,
    num_samples: int,
    max_len: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> List[str]:
    conditions = torch.zeros(num_samples, 1, device=device)
    token_ids = model.sample(
        num_samples=num_samples,
        conditions=conditions,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in token_ids.cpu()]


def compute_metrics(samples: List[str], train_set: set) -> Dict[str, float]:
    unique = set(samples)
    valid = [s for s in samples if to_rdkit(s)]
    novelty_set = [s for s in unique if s not in train_set] if train_set else []
    metrics = {
        "num_samples": len(samples),
        "num_valid": len(valid),
        "num_unique": len(unique),
        "num_novel": len(novelty_set),
        "validity": len(valid) / len(samples) if samples else 0.0,
        "uniqueness": len(unique) / len(samples) if samples else 0.0,
        "novelty": len(novelty_set) / len(unique) if unique else 0.0,
    }
    return metrics


def save_samples(samples: List[str], train_set: set, output_path: Path):
    seen = set()
    records = []
    for idx, s in enumerate(samples):
        mol = to_rdkit(s)
        is_valid = mol is not None
        is_unique = s not in seen
        in_training = s in train_set if train_set else False
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
    model, tokenizer = load_model(args, device)

    train_set = set()
    if args.data_csv is not None:
        df = pd.read_csv(args.data_csv)
        if args.data_col not in df.columns:
            raise ValueError(f"Column '{args.data_col}' not found in {args.data_csv}")
        train_set = set(df[args.data_col].astype(str))

    samples = sample_smiles(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
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
