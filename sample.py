import sys, torch
from pathlib import Path
repo = Path("/Users/tangren/Documents/PolymersGenerator")
sys.path.append(str(repo / "src"))  # 允许导入 src 包
# 导入模块与设备
from src.tokenizer import PolyBertTokenizer
from src.dataset import make_loader
from src.modelv2 import VAESmiles
from src.train import train_one_epoch, val_loss, set_seed
from transformers import AutoModel
import torch.optim as optim
import tqdm as notebook_tqdm

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")

ckpt_path = repo / "checkpoints/modelv2_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)

tokenizer = PolyBertTokenizer("./polybert")
polybert = AutoModel.from_pretrained("./polybert").to(device)

# 加载数据与tokenizer
csv_path = "data/PSMILES_Tg_only.csv"
tokenizer = PolyBertTokenizer("./polybert")
train_loader = make_loader(
    csv_path,
    tokenizer,
    batch_size=128,
    shuffle=True,
    col="PSMILES",
    max_len=256,
)
val_loader = make_loader(
    csv_path,
    tokenizer,
    batch_size=128,
    shuffle=False,
    col="PSMILES",
    max_len=256,
)

model = VAESmiles(
    vocab_size=tokenizer.vocab_size,
    emb_dim=256,
    encoder_hid_dim=polybert.config.hidden_size,
    decoder_hid_dim=512,
    z_dim=128,
    n_layers=1,
    pad_id=tokenizer.pad_id,
    bos_id=tokenizer.bos_id,
    eos_id=tokenizer.eos_id,
    drop=0.1,
    use_polybert=True,
    polybert=polybert,
    freeze_polybert=True,
    polybert_pooling="cls",
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

kl_w = 1.0  # 评估时通常直接用 1
val_loss_value = val_loss(model, val_loader, kl_w, tokenizer.pad_id, device)
print(f"modelv2 验证集损失: {val_loss_value:.4f}")