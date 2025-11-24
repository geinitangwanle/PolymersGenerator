import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoModel

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_ROOT.parent.parent  # .../PolymersGenerator
sys.path.append(str(PROJ_ROOT / "src"))

from dataset_tg import TgStats
from modelv3 import ConditionalVAESmiles
from tokenizer import PolyBertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sample Tg-conditional SMILES and save outputs.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to modelv3 checkpoint.") # 预训练模型检查点路径
    parser.add_argument("--polybert-dir", type=Path, default=Path("./polybert")) # polyBERT 模型目录
    parser.add_argument("--num-per-target", type=int, default=128, help="Samples per Tg target.") # 每个 Tg 目标的生成样本数量
    parser.add_argument("--target-tg", type=float, nargs="+", required=True, help="Target Tg values (Kelvin).") # 目标 Tg 数值列表（开尔文）
    parser.add_argument("--max-len", type=int, default=256) # 生成序列的最大长度
    parser.add_argument("--temperature", type=float, default=1.0) # 采样温度
    parser.add_argument("--top-k", type=int, default=None) # top-k 采样参数
    parser.add_argument("--top-p", type=float, default=None) # top-p 采样参数
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_tg")) # 输出目录
    parser.add_argument("--samples-file", type=str, default="samples_tg.csv") # 生成样本保存文件名
    parser.add_argument("--metrics-file", type=str, default="metrics_tg.json") # 生成指标保存文件名
    return parser.parse_args()
"""python sample_tg.py --checkpoint checkpoints/modelv3_tg.pt --polybert-dir ./polybert --target-tg 350 450 500 --num-per-target 256 --max-len 256 --temperature 0.9 --top-k 50"""

def prepare_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = PolyBertTokenizer(str(args.polybert_dir))
    polybert = AutoModel.from_pretrained(str(args.polybert_dir)).to(device)
    model_kwargs = ckpt["model_kwargs"].copy()
    model_kwargs.update({
        "vocab_size": tokenizer.vocab_size,
        "polybert": polybert,
        "use_polybert": True,
    })
    model = ConditionalVAESmiles(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tg_stats = TgStats(**ckpt["tg_stats"])
    return model, tokenizer, tg_stats


def normalize_targets(targets: List[float], stats: TgStats, device):
    norm = stats.normalize(targets)
    return torch.tensor(norm, dtype=torch.float32, device=device).unsqueeze(-1)


def to_rdkit(smiles: str):
    return Chem.MolFromSmiles(smiles.replace("[*]", "[Xe]"))


def save_samples(records: List[dict], path: Path):
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    device = prepare_device()
    model, tokenizer, tg_stats = load_model(args, device)

    all_records = []
    all_smiles = []
    seen = set()

    for target in args.target_tg:
        cond = normalize_targets([target] * args.num_per_target, tg_stats, device)
        samples = model.sample(
            num_samples=args.num_per_target,
            conditions=cond,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        decoded = [tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in samples.cpu()]
        for idx, smi in enumerate(decoded):
            mol = to_rdkit(smi)
            all_records.append(
                {
                    "target_tg": target,
                    "sample_index": idx,
                    "smiles": smi,
                    "is_valid": mol is not None,
                    "is_unique": smi not in seen,
                }
            )
            seen.add(smi)
        all_smiles.extend(decoded)

    valid = [s for s in all_smiles if to_rdkit(s)]
    unique = set(all_smiles)
    metrics = {
        "num_samples": len(all_smiles),
        "num_valid": len(valid),
        "num_unique": len(unique),
        "validity": len(valid) / len(all_smiles) if all_smiles else 0.0,
        "uniqueness": len(unique) / len(all_smiles) if all_smiles else 0.0,
    }

    samples_path = args.output_dir / args.samples_file
    metrics_path = args.output_dir / args.metrics_file
    save_samples(all_records, samples_path)
    save_metrics(metrics, metrics_path)

    print(f"Saved samples to {samples_path}")
    print(f"Metrics: {json.dumps(metrics, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
