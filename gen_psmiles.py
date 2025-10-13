#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PSMILES 批量生成脚本

用法示例：
1) 单条条件：
   python gen_psmiles.py --ckpt psmiles-gpt --inst "target_Tg=400.0" --n 128 --safe

2) 多条条件（每行一个 instruction，如 target_Tg=350.0）：
   python gen_psmiles.py --ckpt psmiles-gpt --inst_file cond.txt --n 64 --safe

3) 关闭 RDKit 校验：
   python gen_psmiles.py --ckpt psmiles-gpt --inst "target_Tg=400.0" --n 64 --no_rdkit

输出：默认 gen_psmiles.csv
"""

import argparse
import os
import re
from collections import OrderedDict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

SEP = "|cond|"

# ------------------------
# 基础清洗与自检
# ------------------------
ALLOWED_RE = re.compile(r"[A-Za-z0-9\*\[\]\(\)=#\-\+\/\\\.@%]+")

def sanitize(s: str) -> str:
    """仅保留允许字符的前缀，并去除空白。"""
    s = s.strip()
    m = ALLOWED_RE.match(s)
    if not m:
        return ""
    s = m.group(0)
    s = re.sub(r"\s+", "", s)
    return s

def balance_ok(s: str) -> bool:
    """圆括号配平检查。"""
    c = 0
    for ch in s:
        if ch == '(':
            c += 1
        elif ch == ')':
            c -= 1
            if c < 0:
                return False
    return c == 0

def rings_ok(s: str) -> bool:
    """环号成对的粗检：0..9 以及 %nn 必须出现偶数次。"""
    s2 = re.sub(r"%\d{2}", "R", s)  # 双位环号占位为 'R'
    cnt = {}
    for ch in s2:
        if ch.isdigit() or ch == 'R':
            cnt[ch] = cnt.get(ch, 0) + 1
    return all(v % 2 == 0 for v in cnt.values())

# ------------------------
# 模型加载
# ------------------------
def load_model(ckpt, device):
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device).eval()
    # 保险：确保有 pad/eos
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok, model

# ------------------------
# 生成
# ------------------------
def generate_batch(tok, model, instruction, n=64, max_new_tokens=128,
                   top_p=0.9, temperature=0.8, repetition_penalty=1.1,
                   device="cuda", rdkit_check=True, safe=False):
    """
    safe=False: 采样（多样性更高）
    safe=True : 保守解码（beam）+ 清洗自检（有效率更高）
    """
    prompt = instruction.strip() + SEP
    batch_prompts = [prompt] * n
    inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        if safe:
            outs = model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 64),
                do_sample=False,
                num_beams=5,
                length_penalty=0.8,
                repetition_penalty=1.15,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        else:
            outs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

    texts = tok.batch_decode(outs, skip_special_tokens=True)
    gens = [t.split(SEP, 1)[-1].strip() for t in texts]
    gens = [g for g in gens if g]
    gens = list(OrderedDict.fromkeys(gens))  # 去重保序

    # 安全模式：白名单截断 + 括号/环号自检
    if safe:
        cleaned = []
        for s in gens:
            s = sanitize(s)
            if not s: 
                continue
            if not balance_ok(s): 
                continue
            if not rings_ok(s): 
                continue
            cleaned.append(s)
        gens = cleaned

    # 可选：RDKit 粗验（注意 PSMILES 与 SMILES 可能不完全一致）
    if rdkit_check:
        try:
            from rdkit import Chem
            gens = [s for s in gens if Chem.MolFromSmiles(s) is not None]
        except Exception:
            # RDKit 不可用则跳过
            pass

    return gens

# ------------------------
# 主函数
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="微调模型目录，如 psmiles-gpt")
    parser.add_argument("--inst", type=str, default="", help="单条条件（建议形如 target_Tg=400.0）")
    parser.add_argument("--inst_file", type=str, default="", help="多条条件的 txt 文件（每行一条）")
    parser.add_argument("--n", type=int, default=64, help="每条条件生成候选数")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--out", type=str, default="gen_psmiles.csv")
    parser.add_argument("--no_rdkit", action="store_true", help="不使用 RDKit 粗校验")
    parser.add_argument("--safe", action="store_true", help="保守解码 + 清洗自检（提升有效率）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok, model = load_model(args.ckpt, device)

    # 收集指令
    instructions = []
    if args.inst:
        instructions.append(args.inst.strip())
    if args.inst_file:
        with open(args.inst_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instructions.append(line)

    if not instructions:
        raise ValueError("请通过 --inst 或 --inst_file 提供至少一条条件（如：target_Tg=400.0）。")

    rows = []
    for idx, inst in enumerate(instructions, 1):
        gens = generate_batch(
            tok, model, inst,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            device=device,
            rdkit_check=not args.no_rdkit,
            safe=args.safe
        )
        for s in gens:
            rows.append({"instruction": inst, "psmiles": s})
        print(f"[{idx}/{len(instructions)}] {inst} -> 有效候选 {len(gens)} 条")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8")
    print(f"已保存：{args.out}（共 {len(rows)} 条）")

if __name__ == "__main__":
    main()
