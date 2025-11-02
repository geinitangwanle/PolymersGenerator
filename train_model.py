import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# === 超算路径 ===
repo = Path("/share/home/u23514/HTY/PolymersGenerator")
sys.path.append(str(repo / "src"))

# === 导入自定义模块 ===
from src.tokenizer import PolyBertTokenizer
from src.dataset import make_loader
from src.modelv2 import VAESmiles
from src.train import train_one_epoch, val_loss, set_seed

# ==============================
# 核心函数：支持单机多卡DDP训练
# ==============================
def main(rank, world_size):
    # 初始化分布式环境
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    set_seed(42 + rank)

    # === 设备与路径 ===
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] Using device: {device}")

    csv_path = "data/PSMILES_Tg_only.csv"
    tokenizer = PolyBertTokenizer("./polybert")

    # === DataLoader优化 ===
    train_loader = make_loader(
        csv_path, tokenizer,
        batch_size=128,
        shuffle=True,
        col="PSMILES",
        max_len=256,
        num_workers=8,
        pin_memory=True,
        distributed=(world_size > 1),
    )
    val_loader = make_loader(
        csv_path, tokenizer,
        batch_size=128,
        shuffle=False,
        col="PSMILES",
        max_len=256,
        num_workers=8,
        pin_memory=True,
        distributed=(world_size > 1),
    )

    # === 模型加载 ===
    polybert = AutoModel.from_pretrained("./polybert").to(device)
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

    # === 多GPU封装 ===
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # === 优化器与调度器 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = len(train_loader) * 10  # 10 epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
    )

    # === 混合精度训练 ===
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # === 训练循环 ===
    best = float("inf")
    for epoch in range(10):
        kl_w = min(1.0, (epoch + 1) / 10.0)
        train_loss_val = train_one_epoch(
            model, train_loader, optimizer, kl_w, tokenizer.pad_id,
            device, scaler=scaler, scheduler=scheduler
        )

        val_loss_val = val_loss(model, val_loader, kl_w, tokenizer.pad_id, device)
        if rank == 0:
            print(f"[Epoch {epoch+1}/10] train={train_loss_val:.4f} val={val_loss_val:.4f} kl_w={kl_w:.2f}")

            # === 自动保存最优模型 ===
            if val_loss_val + 1e-3 < best:
                best = val_loss_val
                ckpt_dir = repo / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "tokenizer": tokenizer.get_vocab(),
                        "best_val_loss": best,
                    },
                    ckpt_dir / "modelv2_best.pt",
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 检测GPU数量并自动选择单卡/多卡模式
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        mp.spawn(main, args=(n_gpus,), nprocs=n_gpus, join=True)
    else:
        main(rank=0, world_size=1)
