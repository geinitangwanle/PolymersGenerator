#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练一个面向 P(SMILES) 的分词器（化学单元优先 + BPE）。
输入：train.json / val.json（每行：{"instruction": "...", "output": "..."}）
输出：psmiles_tokenizer/ 目录，可被 AutoTokenizer 直接加载。
"""

import os, json

# =========================
# 🔧 配置区（你主要改这里）
# =========================
OUT_DIR = "psmiles_tokenizer"                 # 输出目录
CORPUS_FILES = ["train.json", "val.json"]     # 训练语料（行级 JSON）

# 词表规模与词频阈值
VOCAB_SIZE = 4000
MIN_FREQUENCY = 2

# 句末自动加 <eos>（建议 True，有助于推理早停）
ADD_EOS_AT_END = True

# tokenizers 的 normalizer（不要 lower/strip accents，保持大小写）
USE_NFD_NORMALIZER = True

# special tokens
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]

# 你的分隔符（和训练/推理保持一致）
COND_SEP = "|cond|"

# 化学优先切分（注意：这里使用**字符串**，不是 re.compile）
CHEM_RE = (
    r"(" + COND_SEP.replace("|", r"\|") + r")"  # 分隔符，转义竖线
    r"|(\[[^\]\s]+\])"         # 方括号原子（如 [*], [SiH2]）
    r"|(%\d{2})"               # 双位环号 %10
    r"|([A-Z][a-z]?)"          # 元素（Cl, Br, Si, Ge, ...）
    r"|([cnospBNOFPSI])"       # 芳香/常见元素小写
    r"|([=#/\]\\\(\)\-\+\.@])" # 键与控制符（注意反斜杠转义）
    r"|(\d)"                   # 单位环号 0..9
)

# 保存到 tokenizer_config.json 的配置（供 HF 读取）
MODEL_MAX_LENGTH = 2048
PADDING_SIDE = "right"
# =========================


# ============= 实现部分（一般不用改） =============
from tokenizers import Tokenizer, Regex
from tokenizers.normalizers import Sequence, NFD
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def make_line(ex: dict) -> str:
    """和训练一致：instruction + SEP + output"""
    return f'{ex["instruction"]}{COND_SEP}{ex["output"]}'


def iter_corpus(files):
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                yield make_line(ex)


def main(USE_UNIGRAM: bool = False):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 初始化 tokenizer 模型 + 规范化 + 预分词
    if USE_UNIGRAM:
        raise NotImplementedError("如需 Unigram，请把上方 import 改为 Unigram/UnigramTrainer 并在此处实例化。")
        # tok = Tokenizer(Unigram())
        # trainer = UnigramTrainer(
        #     vocab_size=VOCAB_SIZE,
        #     special_tokens=SPECIAL_TOKENS
        # )
    else:
        tok = Tokenizer(BPE(unk_token="<unk>"))
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY,
            show_progress=True,
            special_tokens=SPECIAL_TOKENS
        )

    if USE_NFD_NORMALIZER:
        tok.normalizer = Sequence([NFD()])   # 轻度规范；不做 lower/去重音

    # ✅ 稳定的预分词器：按化学单元切分
    tok.pre_tokenizer = Split(Regex(CHEM_RE), behavior="isolated")

    # 2) 训练
    tok.train_from_iterator(iter_corpus(CORPUS_FILES), trainer=trainer)

    # 3) 句末自动加 <eos>
    if ADD_EOS_AT_END:
        tok.post_processor = TemplateProcessing(
            single="$0 <eos>",
            special_tokens=[("<eos>", tok.token_to_id("<eos>"))]
        )

    # 4) 保存为 HF 可读目录
    tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
    with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_max_length": MODEL_MAX_LENGTH,
            "padding_side": PADDING_SIDE,
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "bos_token": "<bos>"
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Tokenizer saved to: {OUT_DIR}/")
    print(f"  - vocab_size (actual): {tok.get_vocab_size()}")

    # 简单可视化一下切分
    demo = "[*]C(=O)Cl" + COND_SEP + "[*]C#CC"
    print("Demo pre_tokenize:", tok.pre_tokenizer.pre_tokenize_str(demo))


if __name__ == "__main__":
    # 通过入参传 flag，避免作用域导致的 NameError
    main(USE_UNIGRAM=False)
