## PSMILES 生成器

- 代码目前在 `src/`（`modelv4*.py`, `tokenizer.py`, `dataset_tg.py` 等），通过 `unified_cli.py` 分发到 `scripts/train` 与 `scripts/sample`。
- 训练入口示例：
  - 预训练：`python unified_cli.py train --version v4 --mode pretrain -- --csv data/PI1M_v2_psmiles.csv`
  - Tg 微调：`python unified_cli.py train --version v4 --mode finetune -- --csv data/PSMILES_Tg_only.csv`
- 采样入口示例：
  - 无条件：`python unified_cli.py sample --version v4 --mode uncond -- --checkpoint checkpoints/pretrain_modelv4.pt`
  - Tg 条件：`python unified_cli.py sample --version v4 --mode tg -- --checkpoint checkpoints/finetune_tg_modelv4.pt`
- 可视化：`scripts/plot_latent_v4.py` 支持导出并绘制 latent PCA/直方图。

计划后续将核心代码迁移到 `models/psmiles_generator/` 包装成独立模块（数据集、模型、采样器、评价、可视化）。
