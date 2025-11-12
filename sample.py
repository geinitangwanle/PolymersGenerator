import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel

repo = Path(__file__).resolve().parent
sys.path.append(str(repo / "src"))

from src.tokenizer import PolyBertTokenizer  # noqa: E402
from src.modelv2 import VAESmiles  # noqa: E402
from src.train import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate polymer SMILES samples and save the results.")
    parser.add_argument("--checkpoint", type=Path, default=repo / "checkpoints/modelv2_best.pt",
                        help="Path to the trained model checkpoint.")
    parser.add_argument("--polybert-dir", type=Path, default=repo / "polybert",
                        help="Directory containing the polyBERT weights/tokenizer.")
    parser.add_argument("--data-csv", type=Path, default=repo / "data/PSMILES_Tg_only.csv",
                        help="CSV file used during training (for novelty metric).")
    parser.add_argument("--data-col", type=str, default="PSMILES",
                        help="Column name in the CSV that holds SMILES strings.")
    parser.add_argument("--num-samples", type=int, default=512,
                        help="Number of SMILES to sample.")
    parser.add_argument("--max-len", type=int, default=256,
                        help="Maximum decoding length when sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs",
                        help="Directory to store sampled SMILES and metrics.")
    parser.add_argument("--samples-file", type=str, default="sampled_smiles.csv",
                        help="Filename for the sampled SMILES CSV.")
    parser.add_argument("--metrics-file", type=str, default="sample_metrics.json",
                        help="Filename for the metrics JSON.")
    return parser.parse_args()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args, device) -> Tuple[VAESmiles, PolyBertTokenizer]:
    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = PolyBertTokenizer(str(args.polybert_dir))
    polybert = AutoModel.from_pretrained(str(args.polybert_dir)).to(device)

    model = VAESmiles(
        vocab_size=tokenizer.vocab_size,
        emb_dim=256,
        encoder_hid_dim=polybert.config.hidden_size,
        decoder_hid_dim=512,
        z_dim=128,
        n_layers=1,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        drop=0.1,
        use_polybert=True,
        polybert=polybert,
        freeze_polybert=True,
        polybert_pooling="cls",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def to_rdkit(smiles: str):
    # Replace polymer attachment placeholder with a neutral atom to make RDKit happy.
    return Chem.MolFromSmiles(smiles.replace("[*]", "[Xe]"))


@torch.no_grad()
def sample_smiles(model: VAESmiles, tokenizer: PolyBertTokenizer, device: torch.device,
                  num_samples: int, max_len: int) -> List[str]:
    latent_dim = model.mu.out_features
    z = torch.randn(num_samples, latent_dim, device=device)
    token_ids = model.sample(z, max_len=max_len)
    return [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in token_ids.cpu()]


def compute_metrics(samples: List[str], train_set: set) -> Dict[str, float]:
    unique = set(samples)
    valid = [s for s in samples if to_rdkit(s)]
    novelty_set = [s for s in unique if s not in train_set]
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
    model, tokenizer = load_model(args, device)

    df = pd.read_csv(args.data_csv)
    if args.data_col not in df.columns:
        raise ValueError(f"Column '{args.data_col}' not found in {args.data_csv}")

    train_set = set(df[args.data_col].astype(str))
    samples = sample_smiles(model, tokenizer, device, args.num_samples, args.max_len)

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
