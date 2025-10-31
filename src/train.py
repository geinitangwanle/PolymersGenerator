#============================导入模块================================
import os, torch, random
import numpy as np
from tqdm import tqdm
from dataset import make_loader
from model import VAESmiles
from tokenizer import PolyBertTokenizer
from transformers import AutoModel
#====================================================================
# 设置随机种子，以保证实验可复现
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
# KL 散度损失
def kld_loss(mu, logvar):
    # 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# 训练一个 epoch
def train_one_epoch(model, loader, opt, kl_weight, pad_id, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device) # [B, T]
        decoder_input_ids = batch["decoder_input_ids"].to(device) # [B, T-1]
        labels = batch["labels"].to(device) # [B, T-1]
        attention_mask = batch["attention_mask"].to(device) # [B, T]

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

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
    tokenizer = PolyBertTokenizer(polybert_name)

    # 2) DataLoader
    train_loader = make_loader(
        "data/molecules.csv",
        tokenizer,
        batch_size=256,
        shuffle=True,
        col="smiles",
        max_len=256,
    )
    # 若有独立验证集，可替换为 make_loader("data/val.csv", tokenizer, shuffle=False)
    val_loader = train_loader

    # 3) 模型：使用 polyBERT 编码器 + GRU 解码器
    polybert_encoder = AutoModel.from_pretrained(polybert_name)
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
        freeze_polybert=True,
        polybert_pooling="cls",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

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

if __name__ == "__main__":
    main()
