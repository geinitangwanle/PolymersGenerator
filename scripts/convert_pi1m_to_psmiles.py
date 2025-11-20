#!/usr/bin/env python
"""
Convert PI1M SMILES strings into PolySMILES-friendly form.

The PI1M dataset encodes polymer attachment points with bare `*` characters.
PolyBERT and the rest of this project, however, expect `[ * ]` style wildcard
atoms (i.e. the `*` atom enclosed in square brackets).  This script rewrites
those wildcards while leaving the rest of each string unchanged.

将PI1M SMILES字符串转换为polymiles友好的形式。

PI1M数据集用裸“*”字符编码聚合物附属点。
然而，PolyBERT和这个项目的其余部分使用`[*]`样式通配符
原子（即方括号中的“*”原子）。这个脚本重写了
这些通配符，而每个字符串的其余部分保持不变。

Example usage:

    python scripts/convert_pi1m_to_psmiles.py \
        --input data/PI1M_v2.csv \
        --output data/PI1M_v2_psmiles.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def convert_star_atoms(smiles: str) -> str:
    """字符串层面的 * → [*] 转换"""
    if pd.isna(smiles): # 如果 smiles 是 NaN（例如 DataFrame 里是空缺值），直接返回，不处理
        return smiles

    smiles = smiles.strip()
    if not smiles:
        return smiles

    converted: list[str] = []
    inside_brackets = False
    for ch in smiles:
        if ch == "[":
            inside_brackets = True
            converted.append(ch)
        elif ch == "]":
            inside_brackets = False
            converted.append(ch)
        elif ch == "*" and not inside_brackets:
            converted.append("[*]")
        else:
            converted.append(ch)
    return "".join(converted)


def convert_dataframe(df: pd.DataFrame, *, input_col: str, output_col: str) -> pd.DataFrame:
    """Return copy of df with converted SMILES stored in output_col."""
    if input_col not in df.columns:
        raise ValueError(f"Column '{input_col}' not found in input CSV.")

    df = df.copy()
    df[output_col] = df[input_col].map(convert_star_atoms)

    if output_col == input_col:
        return df

    reordered_cols: list[str] = [output_col] + [c for c in df.columns if c != output_col]
    return df[reordered_cols]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/PI1M_v2.csv"),
        help="Input CSV containing PI1M SMILES (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/PI1M_v2_psmiles.csv"),
        help="Destination CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES",
        help="Column name containing raw PI1M SMILES (default: %(default)s)",
    )
    parser.add_argument(
        "--output-col",
        type=str,
        default="PSMILES",
        help="Name for the converted column (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    converted = convert_dataframe(df, input_col=args.smiles_col, output_col=args.output_col)
    converted.to_csv(args.output, index=False)

    total = len(df)
    converted_count = converted[args.output_col].notna().sum()
    print(
        f"Converted {converted_count}/{total} rows from '{args.smiles_col}' "
        f"and wrote result to {args.output}"
    )


if __name__ == "__main__":
    main()
