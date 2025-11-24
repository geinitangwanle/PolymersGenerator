"""
统一入口：通过一个脚本分发到 model v1/v2/v3/v4 的训练和采样脚本。
使用方式示例：
  - 训练 v4 预训练：python unified_cli.py train --version v4 --mode pretrain -- --csv data/PI1M_v2_psmiles.csv
  - 训练 v4 Tg 微调：python unified_cli.py train --version v4 --mode finetune -- --csv data/PSMILES_Tg_only.csv
  - 训练 v3 Tg：    python unified_cli.py train --version v3 -- --csv data/PSMILES_Tg_only.csv
  - 训练 v1/v2：    python unified_cli.py train --version v2
  - 采样 v4 预训练：python unified_cli.py sample --version v4 --mode uncond -- --checkpoint checkpoints/pretrain_modelv4.pt
  - 采样 v4 Tg：    python unified_cli.py sample --version v4 --mode tg -- --checkpoint checkpoints/finetune_tg_modelv4.pt
  - 采样 v3 Tg：    python unified_cli.py sample --version v3 -- --checkpoint checkpoints/modelv3_tg.pt
  - 采样 v1/v2：    python unified_cli.py sample --version v2 -- --checkpoint checkpoints/modelv2_best.pt

额外参数用 `--` 传递，原脚本会按自己的 argparse 接收。
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent


def run_script(script: Path, extra_args: List[str], *, prepend: Optional[List[str]] = None):
    cmd = [sys.executable, str(script)]
    if prepend:
        cmd.extend(prepend)
    cmd.extend(extra_args)
    print(f"[unified-cli] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def dispatch_train(args):
    extra = args.extra or []
    version = args.version.lower()
    mode = args.mode.lower() if args.mode else None

    if version == "v1":
        run_script(ROOT / "scripts" / "train" / "train_v1_base.py", extra)
    elif version == "v2":
        run_script(ROOT / "scripts" / "train" / "train_v2_base.py", extra)
    elif version == "v3":
        run_script(ROOT / "scripts" / "train" / "train_v3_tg.py", extra)
    elif version == "v4":
        if mode is None:
            raise SystemExit("v4 训练需要指定 --mode pretrain 或 --mode finetune")
        if mode == "pretrain":
            run_script(ROOT / "scripts" / "train" / "train_v4_pretrain.py", extra)
        elif mode in {"finetune", "finetune_tg"}:
            run_script(ROOT / "scripts" / "train" / "train_v4_finetune.py", extra)
        else:
            raise SystemExit(f"未知的 v4 训练模式: {mode}")
    else:
        raise SystemExit(f"不支持的版本: {version}")


def dispatch_sample(args):
    extra = args.extra or []
    version = args.version.lower()
    mode = args.mode.lower() if args.mode else None

    if version in {"v1", "v2"}:
        prepend = ["--model-version", version]
        run_script(ROOT / "scripts" / "sample" / "sample_v2_base.py", extra, prepend=prepend)
    elif version == "v3":
        run_script(ROOT / "scripts" / "sample" / "sample_v3_tg.py", extra)
    elif version == "v4":
        if mode is None:
            mode = "uncond"
        if mode in {"uncond", "pretrain"}:
            run_script(ROOT / "scripts" / "sample" / "sample_v4_uncond.py", extra)
        elif mode in {"tg", "finetune"}:
            run_script(ROOT / "scripts" / "sample" / "sample_v4_mask.py", extra)
        else:
            raise SystemExit(f"未知的 v4 采样模式: {mode}")
    else:
        raise SystemExit(f"不支持的版本: {version}")


def build_parser():
    parser = argparse.ArgumentParser(description="Unified CLI dispatcher for model v1/v2/v3/v4 training and sampling.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a model version.")
    train_p.add_argument("--version", required=True, choices=["v1", "v2", "v3", "v4"], help="Model version.")
    train_p.add_argument("--mode", help="Training mode (v4: pretrain/finetune; others忽略)")
    train_p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to the underlying script.")
    train_p.set_defaults(func=dispatch_train)

    sample_p = sub.add_parser("sample", help="Sample from a trained checkpoint.")
    sample_p.add_argument("--version", required=True, choices=["v1", "v2", "v3", "v4"], help="Model version.")
    sample_p.add_argument("--mode", help="Sampling mode (v4: uncond/tg; others忽略)")
    sample_p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to the underlying script.")
    sample_p.set_defaults(func=dispatch_sample)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
