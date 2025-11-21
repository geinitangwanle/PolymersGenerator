import torch
from typing import Dict, Iterable, List, Optional, Set


class SyntaxMasker:
    """
    A lightweight SMILES grammar helper that provides token-level masks during sampling.
    它在采样生成 SMILES 时，根据“前缀序列”决定“下一个 token 合不合法”，返回一个 bool mask。
    1 表示“允许采样这个 token”，0 表示“禁用”。
    它是“保守的”：只禁止明显很不合法的续写，比如：
    - 一上来就 ')'
    - 连续出现键（bond）符号 "--"、"=="
    但不会做特别严格的语法检查，避免过度约束模型。
    """

    def __init__(self, tokenizer, vocab_size: Optional[int] = None):
        # 支持传入 PolyBertTokenizer 或直接的 HF tokenizer
        self.tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        base_vocab_size = self.tokenizer.vocab_size
        # 允许外部指定 vocab_size（以防模型输出维度与 tokenizer.vocab_size 不一致）
        self.vocab_size = int(vocab_size) if vocab_size is not None else base_vocab_size

        self.pad_id = self._safe_get_id(tokenizer, "pad_token_id") # 取出 pad_id
        self.bos_id = self._safe_get_id(tokenizer, "bos_token_id", fallback_attr="cls_token_id") # 取出 bos_id
        self.eos_id = self._safe_get_id(tokenizer, "eos_token_id", fallback_attr="sep_token_id") # 取出 eos_id

        # 逐 id 分类，便于后续快速取掩码
        self.token_type: Dict[int, str] = {} # 每个 token id 对应一个类别字符串
        self.atom_ids: Set[int] = set() # 原子或原子片段 token
        self.bond_ids: Set[int] = set() # 键（bond）符号 token（- = # : / \ .）
        self.branch_open_ids: Set[int] = set() # 分支开括号 "(" token
        self.branch_close_ids: Set[int] = set() # 分支闭括号 ")" token
        self.ring_ids: Set[int] = set() # 环编号 1 2 3 ...
        self.bracket_open_ids: Set[int] = set() # 方括号 "[" token
        self.bracket_close_ids: Set[int] = set() # 方括号 "]" token

        # 遍历词表，分类每个 token
        for tid in range(self.vocab_size): 
            tok = self._safe_convert_id(tid) # 将 id 转为 token 字符串
            category = self._classify_token(tid, tok) # 用 _classify_token 判断这个 token 属于哪种类型
            self.token_type[tid] = category
            if category == "atom":
                self.atom_ids.add(tid)
            elif category == "bond":
                self.bond_ids.add(tid)
            elif category == "branch_open":
                self.branch_open_ids.add(tid)
            elif category == "branch_close":
                self.branch_close_ids.add(tid)
            elif category == "ring":
                self.ring_ids.add(tid)
            elif category == "bracket_open":
                self.bracket_open_ids.add(tid)
            elif category == "bracket_close":
                self.bracket_close_ids.add(tid)

        # 备用：若未显式识别到括号标记，仍保证有基本的 atom 集合
        if not self.atom_ids:
            self.atom_ids = set(range(self.vocab_size))

    def _safe_get_id(self, tokenizer, attr: str, fallback_attr: Optional[str] = None) -> Optional[int]:
        val = getattr(tokenizer, attr, None) # 尝试从 tokenizer 上取 attr，没有的话再尝试 fallback_attr
        if val is None and fallback_attr is not None:
            val = getattr(tokenizer, fallback_attr, None)
        return val

    def _safe_convert_id(self, tid: int) -> str:
        try:
            return self.tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            return f"<unk_{tid}>"

    def _normalize_token(self, tok: str) -> str:
        # 去掉常见的前导标记（BPE/SentencePiece）
        return tok.lstrip("Ġ▁").strip()

    def _classify_token(self, tid: int, tok: str) -> str:
        if tid in {self.pad_id, self.bos_id, self.eos_id}:
            return "special"
        norm = self._normalize_token(tok)
        if norm == "(":
            return "branch_open"
        if norm == ")":
            return "branch_close"
        if norm == "[":
            return "bracket_open"
        if norm == "]":
            return "bracket_close"
        if norm in {"-", "=", "#", ":", "/", "\\", "."}:
            return "bond"
        if norm in set("123456789"):
            return "ring"
        # 以 "[" 开头且包含 "]" 视作完整 bracket 原子（最常见的子词形式）
        if norm.startswith("[") and "]" in norm:
            return "atom"
        # 其它默认当作原子/片段，避免过度屏蔽
        return "atom"

    # 扫描前缀，返回当前状态
    def _roll_state(self, prefix_ids: Iterable[int]):
        paren_depth = 0 # 当前分支括号 ( / ) 的嵌套深度
        in_bracket = False # 当前是否在方括号 [] 内
        last_type = "start" # 最后一个 token 的类别
        seen = 0 # 统计实际看到的 token 个数（排除 BOS 等）
        for tid in prefix_ids: # 遍历前缀 token id 列表
            if tid == self.bos_id: # 跳过 BOS
                continue
            ttype = self.token_type.get(tid, "atom")  # 查 token_type，找不到的话默认 "atom"
            seen += 1
            if ttype == "branch_open": # 遇到 '('，嵌套深度加 1
                paren_depth += 1
            elif ttype == "branch_close": # 遇到 ')'，嵌套深度减 1（但不小于 0）
                paren_depth = max(0, paren_depth - 1)
            elif ttype == "bracket_open": # 遇到 '['，标记进入方括号
                in_bracket = True
            elif ttype == "bracket_close": # 遇到 ']'，标记离开方括号
                in_bracket = False
            last_type = ttype
        start = seen == 0 # 若一个非 BOS token 都没看到，则说明还在序列开头。
        return {
            "start": start, # 是否在序列开头
            "paren_depth": paren_depth, # 当前分支括号嵌套深度
            "in_bracket": in_bracket, # 当前是否在方括号内
            "last": last_type, # 最后一个 token 的类别
        }

    # 根据前缀 ids，返回允许的下一个 token mask
    def allowed_mask(self, prefix_ids: List[int], device=None) -> torch.BoolTensor:
        state = self._roll_state(prefix_ids) # （开始/括号深度/是否在 bracket 里/上一个 token 类型）
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device) # 初始化全 0 的 mask，初始化为全 False。
        allowed: Set[int] = set() # 用一个 set 临时记录合法的 token id，最后再写回 mask

        # 基础允许集
        atom_like = self.atom_ids | self.bracket_open_ids | self.bracket_close_ids # 基础允许集：原子类 + 方括号开闭

        if state["start"]: # 开头时，只允许原子类和方括号，不允许一上来就出现 )、bond、数字等
            allowed |= atom_like 
        else:
            last = state["last"] # 上一个 token 的类别
            if last in {"atom", "bracket_close", "ring", "branch_close"}: # 上一个是原子类、']'、数字或 ')' 时
                allowed |= atom_like # 允许原子类
                allowed |= self.branch_open_ids # 允许 '('
                allowed |= self.bond_ids # 允许 bond 符号
                allowed |= self.ring_ids # 允许数字环标记
            elif last == "bond": # 上一个是 bond 符号时,禁止连续 bond，如 "C==", "C--"
                allowed |= atom_like # 允许原子类
                allowed |= self.ring_ids # 允许数字环标记
            elif last == "branch_open": # 上一个是 '(' 时
                allowed |= atom_like # 允许原子类
                allowed |= self.bond_ids # 允许 bond 符号
            elif last == "bracket_open": # 上一个是 '[' 时
                allowed |= atom_like # 允许原子类
                allowed |= self.bracket_close_ids # 允许 ']'

            # 允许关闭括号
            if state["paren_depth"] > 0 and last not in {"bond", "branch_open"}:
                allowed |= self.branch_close_ids

            # 合法结束条件
            if (
                self.eos_id is not None # eos_id 存在
                and state["paren_depth"] == 0
                and not state["in_bracket"] # 不在方括号内
                and last not in {"start", "bond", "branch_open"}
            ):
                allowed.add(self.eos_id)

        # 永久屏蔽 pad/bos
        for bad in (self.pad_id, self.bos_id):
            if bad is not None and bad in allowed:
                allowed.remove(bad)

        # 若 atom_like 为空，则允许所有 token，防止采样时对 mask 做 softmax 出现“全是 -inf 的情况”
        if not allowed:
            # 兜底：至少允许原子类，避免全 0 导致崩溃
            allowed |= atom_like or set(range(self.vocab_size)) 

        mask[list(allowed)] = True # 将合法 token id 对应位置设为 True
        return mask

    # 对 batch 里的每条序列生成掩码
    def batch_mask(
        self,
        batch_ids: torch.Tensor, # [B, T]
        finished: Optional[torch.Tensor] = None, # 形状 [B] 的 bool / 0/1 向量，标记这个样本是否已经结束（比如 beam search 中一些分支已经采到 EOS）
        device=None,
    ) -> torch.BoolTensor:
        """
        Args:
            batch_ids: [B, T] 当前已生成的 token ids
            finished: [B] 标记样本是否已结束；若结束则仅允许 eos
        """
        masks = []
        for i, row in enumerate(batch_ids):
            pref = row.tolist() # 转为普通列表
            if finished is not None and bool(finished[i]): # 若样本已结束
                mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device) # 全 0 掩码
                if self.eos_id is not None: # 仅允许 eos
                    mask[self.eos_id] = True # 设置 eos 位置为 True
                masks.append(mask) # 添加该掩码到列表
                continue
            masks.append(self.allowed_mask(pref, device=device)) # 计算允许掩码并添加到列表
        return torch.stack(masks, dim=0) 
