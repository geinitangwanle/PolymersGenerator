import torch, pandas as pd
from rdkit import Chem
from model import VAESmiles
from tokenizer import PolyBertTokenizer
from transformers import AutoModel

@torch.no_grad()
def decode_ids(tok, ids):
    return tok.decode(ids.tolist())

def is_valid_smiles(s):
    # 对 pSMILES，若含 * 或 { }，RDKit 可能失败；你可以定义“宽松有效性”
    mol = Chem.MolFromSmiles(s)
    return mol is not None

def main(n_samples=1000, max_len=200):
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
    tokenizer_name = ckpt.get("tokenizer_name", "kuelumbus/polyBERT")
    tokenizer = PolyBertTokenizer(tokenizer_name)
    polybert_encoder = AutoModel.from_pretrained(tokenizer_name)
    model = VAESmiles(
        vocab_size=tokenizer.vocab_size,
        emb_dim=256,
        encoder_hid_dim=polybert_encoder.config.hidden_size,
        decoder_hid_dim=512,
        z_dim=128,
        n_layers=1,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        drop=0.1,
        use_polybert=True,
        polybert=polybert_encoder,
        freeze_polybert=True,
    ).eval()

    model.load_state_dict(ckpt["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    Z = torch.randn(n_samples, 128, device=device)
    ids = model.sample(Z, max_len=max_len)
    gen = [decode_ids(tokenizer, row) for row in ids.cpu()]

    # 评估
    valid = [s for s in gen if is_valid_smiles(s)]
    validity = len(valid) / len(gen)

    # 唯一性/新颖性
    uniq = set(gen)
    uniqueness = len(uniq) / len(gen)

    train = pd.read_csv("data/molecules.csv")["smiles"].astype(str).tolist()
    train_set = set(train)
    novelty = len([s for s in uniq if s not in train_set]) / len(uniq)

    print(f"Validity:   {validity:.3f}")
    print(f"Uniqueness: {uniqueness:.3f}")
    print(f"Novelty:    {novelty:.3f}")

    # 保存生成结果
    pd.DataFrame({"smiles": gen}).to_csv("generated.csv", index=False)
    print("Saved to generated.csv")

if __name__ == "__main__":
    main()
