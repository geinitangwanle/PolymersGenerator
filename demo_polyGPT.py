from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# ========================
# 基础配置
# ========================
MODEL_NAME = "./gpt2_local"            # 或 "openai-community/gpt2"
TOKENIZER_PATH = "./psmiles_tokenizer" # ← 使用你刚训练好的 tokenizer 目录
SEP = "|cond|"                         # 必须与 tokenizer 训练时一致

# ========================
# Tokenizer：确保有 pad_token
# ========================
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

# GPT-2 默认无 pad_token；先补上，再让模型 resize 词表（顺序很重要）
if tok.pad_token is None:
    if tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    else:
        tok.add_special_tokens({"pad_token": "<|pad|>"})

# ========================
# 模型
# ========================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# 根据 tokenizer 大小调整词表（必须在 pad/special tokens 处理之后）
model.resize_token_embeddings(len(tok))

# ========================
# 构造样本
# ========================
def build_example(ex):
    cond = ex["instruction"]
    out  = ex["output"]
    in_ids  = tok(cond + SEP, add_special_tokens=False).input_ids
    out_ids = tok(out,            add_special_tokens=False).input_ids

    # ✅ 手动补一个 <eos>，让模型学会“收尾”
    if tok.eos_token_id is not None:
        out_ids = out_ids + [tok.eos_token_id]

    ex["input_ids"] = in_ids + out_ids
    ex["labels"]    = [-100]*len(in_ids) + out_ids
    return ex


ds = load_dataset("json", data_files={"train": "train.json", "val": "val.json"})
ds = ds.map(build_example, remove_columns=ds["train"].column_names)

# ========================
# Data Collator（padding & mask）
# ========================
def collate(batch):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labs = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=tok.pad_token_id)
    labs = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
    attn = (ids != tok.pad_token_id)
    return {"input_ids": ids, "labels": labs, "attention_mask": attn}

# ========================
# 关键 token 加权 Loss（可选保留）
# ========================
_raw_ids = tok.convert_tokens_to_ids(["[*]", "=", "#", "(", ")", "1", "2", "3"])
KT = set([i for i in _raw_ids if i is not None and i >= 0])

class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kt_ids = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        logits = out.logits
        sl = labels[:, 1:].contiguous()
        lg = logits[:, :-1].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        loss = loss_fct(lg.view(-1, lg.size(-1)), sl.view(-1))

        if self._kt_ids is None:
            self._kt_ids = torch.tensor(sorted(list(KT)), device=sl.device, dtype=sl.dtype)

        with torch.no_grad():
            key_mask = torch.isin(sl.view(-1), self._kt_ids)

        loss = torch.where(key_mask, loss * 1.8, loss).mean()
        return (loss, out) if return_outputs else loss

# ========================
# 训练参数
# ========================
args = TrainingArguments(
    output_dir="psmiles-gpt-1",
    learning_rate=3e-5,
    num_train_epochs=15,
    per_device_train_batch_size=8,   # 单机2卡等效原16时，设为8
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    bf16=True,                       # A800 上建议 BF16
    ddp_find_unused_parameters=False,
    report_to=["tensorboard"],
    logging_dir="runs/psmiles"
)

# ========================
# 训练
# ========================
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    data_collator=collate,
)
trainer.train()

# ========================
# 保存（模型 + tokenizer）
# ========================
trainer.save_model("psmiles-gpt-1")   # 保存权重与配置
tok.save_pretrained("psmiles-gpt-1")  # 保存与之匹配的 tokenizer
