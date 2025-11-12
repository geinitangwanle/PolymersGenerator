#============================导入模块================================
import os, torch, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import make_loader
from model import VAESmiles
from tokenizer import PolyBertTokenizer
from transformers import AutoModel
from contextlib import nullcontext
#====================================================================
# 设置随机种子，以保证实验可复现
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
# KL 散度损失
def kld_loss(mu, logvar):
    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def split_dataframe(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
):
    """Deterministically split dataframe into train/val/test subsets."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be less than 1.")

    n = len(df)
    if n < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_size = max(1, int(round(n * train_frac)))
    val_size = max(1, int(round(n * val_frac)))
    if train_size + val_size >= n:
        val_size = max(1, min(val_size, n - train_size - 1))
    test_size = n - train_size - val_size
    if test_size <= 0:
        raise ValueError("Split produced an empty test set; adjust fractions.")

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )

# polybert微调
def configure_polybert_finetuning(
    polybert,
    *,
    train_last_n_layers: int = 0,
    unfreeze_embedding_layernorm: bool = True,
):
    if polybert is None:
        return []

    for param in polybert.parameters(): # 默认冻结所有参数
        param.requires_grad = False

    if train_last_n_layers <= 0: # 如果设定解冻层数为0，则不微调任何层
        polybert.eval()
        return []

    polybert.train()
    encoder = getattr(polybert, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        layers = encoder.layer
        n_layers = len(layers)
        n_to_train = min(max(1, train_last_n_layers), n_layers)
        for layer in layers[-n_to_train:]:
            for param in layer.parameters(): # 只解冻最后 n 层
                param.requires_grad = True 

    if hasattr(polybert, "pooler"):
        for param in polybert.pooler.parameters():
            param.requires_grad = True

    if unfreeze_embedding_layernorm:
        embeddings = getattr(polybert, "embeddings", None)
        if embeddings is not None:
            layer_norm = getattr(embeddings, "LayerNorm", None) or getattr(embeddings, "layer_norm", None)
            if layer_norm is not None:
                for param in layer_norm.parameters():
                    param.requires_grad = True

    return [p for p in polybert.parameters() if p.requires_grad]


# 训练一个 epoch
def train_one_epoch(model, loader, opt, kl_weight, pad_id, device, *, scaler=None, scheduler=None):
    model.train()
    total = 0.0
    device_type = device.type if isinstance(device, torch.device) else device
    use_amp = scaler is not None and device_type.startswith("cuda")
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device) # [B, T]
        decoder_input_ids = batch["decoder_input_ids"].to(device) # [B, T-1]
        labels = batch["labels"].to(device) # [B, T-1]
        attention_mask = batch["attention_mask"].to(device) # [B, T]

        ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            logits, mu, logvar = model(
                encoder_input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=attention_mask,
            )
            # 交叉熵（忽略 pad）
            loss_rec = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad_id
            )
            loss_kld = kld_loss(mu, logvar) # KL 散度
            loss = loss_rec + kl_weight * loss_kld # 总损失

        opt.zero_grad(set_to_none=True)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * input_ids.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def val_loss(model, loader, kl_weight, pad_id, device):
    model.eval(); total = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits, mu, logvar = model(
            encoder_input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=attention_mask,
        )
        loss_rec = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=pad_id
        )
        loss_kld = kld_loss(mu, logvar)
        total += (loss_rec + kl_weight * loss_kld).item() * input_ids.size(0)
    return total / len(loader.dataset)

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 载入 polyBERT tokenizer
    polybert_name = "kuelumbus/polyBERT"
    polybert_train_last_n = int(os.getenv("POLYBERT_TRAIN_LAST_N", 2)) # 解冻最后2层进行微调
    polybert_lr = float(os.getenv("POLYBERT_LR", 1e-5))
    tokenizer = PolyBertTokenizer(polybert_name)

    # 1.5) 加载数据并划分训练/验证/测试
    data_path = Path("data/molecules.csv")
    df = pd.read_csv(data_path)
    train_df, val_df, test_df = split_dataframe(df, train_frac=0.8, val_frac=0.1, seed=42)

    # 2) DataLoader
    train_loader = make_loader(
        train_df,
        tokenizer,
        batch_size=256,
        shuffle=True,
        col="smiles",
        max_len=256,
    )
    val_loader = make_loader(
        val_df,
        tokenizer,
        batch_size=256,
        shuffle=False,
        col="smiles",
        max_len=256,
    )
    test_loader = make_loader(
        test_df,
        tokenizer,
        batch_size=256,
        shuffle=False,
        col="smiles",
        max_len=256,
    )

    # 3) 模型：使用 polyBERT 编码器
    polybert_encoder = AutoModel.from_pretrained(polybert_name)
    trainable_polybert_params = configure_polybert_finetuning(
        polybert_encoder,
        train_last_n_layers=polybert_train_last_n,
    )
    if trainable_polybert_params:
        trained_params = sum(p.numel() for p in trainable_polybert_params)
        print(f"Fine-tuning last {polybert_train_last_n} polyBERT layers (~{trained_params:,} params).")
    else:
        print("polyBERT kept frozen.")
    bos_token_id = tokenizer.bos_id
    eos_token_id = tokenizer.eos_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Tokenizer must provide CLS/BOS and SEP/EOS token ids for decoding.")
    model = VAESmiles(
        vocab_size=tokenizer.vocab_size,
        emb_dim=256,
        encoder_hid_dim=polybert_encoder.config.hidden_size,
        decoder_hid_dim=512,
        z_dim=128,
        n_layers=1,
        pad_id=tokenizer.pad_id,
        bos_id=bos_token_id,
        eos_id=eos_token_id,
        drop=0.1,
        use_polybert=True,
        polybert=polybert_encoder,
        freeze_polybert=False,
        polybert_pooling="cls",
    ).to(device)
    base_lr = 3e-4
    weight_decay = 0.01
    polybert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("polybert."):
            polybert_params.append(param)
        else:
            other_params.append(param)
    param_groups = [{"params": other_params, "lr": base_lr, "weight_decay": weight_decay}]
    if polybert_params:
        param_groups.append({"params": polybert_params, "lr": polybert_lr, "weight_decay": weight_decay})
    opt = torch.optim.AdamW(param_groups)

    # 4) KL 退火
    epochs = 30
    best = 1e9; patience, bad = 5, 0
    for ep in range(1, epochs+1):
        # 线性退火到 1.0（也可用 sigmoid 退火）
        kl_w = min(1.0, ep / 10.0)

        tr = train_one_epoch(model, train_loader, opt, kl_w, tokenizer.pad_id, device)
        va = val_loss(model, val_loader, kl_w, tokenizer.pad_id, device)
        print(f"[{ep}/{epochs}] train {tr:.4f}  val {va:.4f}  kl_w={kl_w:.2f}")

        if va < best - 1e-3:
            best, bad = va, 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.get_vocab(),
                    "pad_token_id": tokenizer.pad_id,
                    "bos_token_id": tokenizer.bos_id,
                    "eos_token_id": tokenizer.eos_id,
                    "tokenizer_name": polybert_name,
                    "use_polybert": True,
                },
                "checkpoints/best.pt",
            )
        else:
            bad += 1
            if bad >= patience:
                print("Early stop.")
                break

    # 5) 训练结束后在测试集上做一次评估（kl_weight=1.0 更常见）
    test_loss_val = val_loss(model, test_loader, kl_weight=1.0, pad_id=tokenizer.pad_id, device=device)
    print(f"Test loss {test_loss_val:.4f}")

if __name__ == "__main__":
    main()
