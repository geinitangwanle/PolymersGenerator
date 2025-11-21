import torch
import re
from typing import List, Set, Dict, Optional

class SyntaxMasker:
    """
    Minimal-restriction SMILES syntax masker.
    目标：提高有效率，而不强行约束模型。
    
    仅屏蔽 5 类确定非法的续写：
      1) 开头不能是 bond、')'、数字（ring digit）
      2) bond 后不能再次 bond
      3) '(' 后不能 ')'
      4) 不允许出现未闭合 '[' 时生成 '['
      5) 不允许 ']' 出现在没有匹配 '[' 的情况下

    所有其他结构全部允许 —— 让模型自由选择。
    """

    BOND_CHARS = {"-", "=", "#", ":", "/", "\\"}

    def __init__(self, tokenizer, vocab_size: Optional[int] = None):
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.vocab_size = vocab_size or self.tokenizer.vocab_size

        self.pad_id = self._safe_get_id("pad_token_id")
        self.bos_id = self._safe_get_id("bos_token_id", fallback="cls_token_id")
        self.eos_id = self._safe_get_id("eos_token_id", fallback="sep_token_id")

        # 分类所有 token
        self.token_type: Dict[int, str] = {}
        self.ring_digit: Dict[int, str] = {}

        for tid in range(self.vocab_size):
            tok = self._norm(self.tokenizer.convert_ids_to_tokens(tid))

            if re.fullmatch(r"\d", tok):     # 单个数字 ring digit
                self.token_type[tid] = "ring"
                self.ring_digit[tid] = tok
            elif tok in self.BOND_CHARS:
                self.token_type[tid] = "bond"
            elif tok == "(":
                self.token_type[tid] = "open_paren"
            elif tok == ")":
                self.token_type[tid] = "close_paren"
            elif tok == "[":
                self.token_type[tid] = "open_bracket"
            elif tok == "]":
                self.token_type[tid] = "close_bracket"
            else:
                self.token_type[tid] = "atom"  # 默认放宽

    def _safe_get_id(self, attr, fallback=None):
        v = getattr(self.tokenizer, attr, None)
        if v is None and fallback:
            v = getattr(self.tokenizer, fallback, None)
        return v

    def _norm(self, tok: str) -> str:
        return tok.lstrip("Ġ▁").strip()

    # 计算 prefix 状态：是否在 bracket 内，记录最后一个 token 类型
    def _roll_state(self, prefix: List[int]):
        in_bracket = False
        last_type = "start"

        for tid in prefix:
            if tid == self.bos_id:
                continue
            ttype = self.token_type.get(tid, "atom")

            if ttype == "open_bracket":
                in_bracket = True
            elif ttype == "close_bracket":
                in_bracket = False

            last_type = ttype

        is_start = all(tid == self.bos_id for tid in prefix)

        return {
            "in_bracket": in_bracket,
            "last": last_type,
            "start": is_start
        }

    def allowed_mask(self, prefix_ids: List[int], device=None):
        state = self._roll_state(prefix_ids)
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)

        last = state["last"]

        # ---------------------------
        # 1) 开头限制：不能以这些开始
        # ---------------------------
        if state["start"]:
            for tid, typ in self.token_type.items():
                if typ in {"bond", "close_paren", "close_bracket", "ring"}:
                    mask[tid] = False

        # ---------------------------
        # 2) bond 后不能接 bond
        # ---------------------------
        if last == "bond":
            for tid, typ in self.token_type.items():
                if typ == "bond":
                    mask[tid] = False

        # ---------------------------
        # 3) '(' 后不能直接 ')'
        # ---------------------------
        if last == "open_paren":
            for tid, typ in self.token_type.items():
                if typ == "close_paren":
                    mask[tid] = False

        # ---------------------------
        # 4) 方括号规则：类 XML 匹配
        # ---------------------------
        if state["in_bracket"]:
            # 不能生成新的 '['
            for tid, typ in self.token_type.items():
                if typ == "open_bracket":
                    mask[tid] = False
        else:
            # 不在 bracket 内时不能生成 ']'
            for tid, typ in self.token_type.items():
                if typ == "close_bracket":
                    mask[tid] = False

        # ---------------------------
        # 保留 eos 合法结束
        # ---------------------------
        if self.eos_id is not None:
            mask[self.eos_id] = True

        return mask

    def batch_mask(self, batch_ids: torch.Tensor, finished=None, device=None):
        out = []
        B = batch_ids.shape[0]

        for i in range(B):
            if finished is not None and bool(finished[i]):
                m = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
                if self.eos_id is not None:
                    m[self.eos_id] = True
                out.append(m)
                continue

            out.append(self.allowed_mask(batch_ids[i].tolist(), device=device))

        return torch.stack(out, dim=0)
