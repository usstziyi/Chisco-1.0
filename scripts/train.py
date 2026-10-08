"""训练 Chisco-1.0 的 EEG→Text 检索模型（对比学习）。

组合的三个文件:
    encoder/eeg_encoder_tsconv_fixed.py  → TSConvFixedEEGEncoder
    encoder/text_encoder_labse.py        → LaBSETextEncoder（冻结，带文本缓存）
    scripts/dataloader.py                → 单 day 内部划分 train / val

对齐方式:
    EEG  encoder 输出 (B, D)，已 L2 归一化
    LaBSE        输出 (B, D)，已 L2 归一化（LaBSE 的 D = 768）
    两者点积即 cosine similarity。

损失（对称 InfoNCE，CLIP 式）:
    logits = eeg_emb @ text_emb.T / temperature
    loss   = (CE(logits, arange) + CE(logits.T, arange)) / 2

指标:
    val loss、val top-1 / top-5 检索准确率（对整个 val 集做排序）

说明:
    单个 day 的文本实测全部唯一，因此 in-batch 对比不存在
    「重复句子 = 假负样本」问题。LaBSE 冻结且按文本做内存缓存，
    多 epoch 下每条文本只编码一次。

用法:
    uv run python scripts/train.py --subject 01 --task imagine --day 1
    uv run python scripts/train.py --subject 01 --task imagine --day 1 --epochs 50

快速冒烟（只跑少量 batch）:
    uv run python scripts/train.py --subject 01 --task imagine --day 1 --max-batches 3
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Hugging Face 镜像，与 encoder/text_encoder_labse.py 保持一致
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 让 encoder 包可导入。
#
# scripts 目录不必手动加：python scripts/train.py 时，
# Python 已把脚本所在目录放在 sys.path[0]，
# 所以 from dataloader import ... 直接可用。
sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import build_dataloaders
from encoder import LaBSETextEncoder, TSConvFixedEEGEncoder


CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


# ============================================================
# Loss / 指标
# ============================================================

def info_nce(
    eeg_emb: torch.Tensor,
    text_emb: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """对称 InfoNCE。第 i 条 EEG 对应第 i 条文本。"""
    logits = eeg_emb @ text_emb.t() / temperature
    labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def in_batch_top1(eeg_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
    labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)
    return (eeg_emb @ text_emb.t()).argmax(dim=1).eq(labels).float().mean().item()


# ============================================================
# 训练 / 验证
# ============================================================

def train_one_epoch(
    eeg_encoder, loader, text_encoder, optimizer, device, temperature, max_batches,
) -> dict:
    eeg_encoder.train()
    total_loss = 0.0
    total_top1 = 0.0
    seen = 0
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        text_emb = text_encoder.encode_unique(batch["text"])
        eeg_emb = eeg_encoder(batch["eeg"].to(device, non_blocking=True))
        loss = info_nce(eeg_emb, text_emb, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["text"])
        total_top1 += in_batch_top1(eeg_emb.detach(), text_emb) * len(batch["text"])
        seen += len(batch["text"])
    return {"loss": total_loss / seen, "top1": total_top1 / seen, "n": seen}


@torch.no_grad()
def evaluate(eeg_encoder, loader, text_encoder, device, temperature, max_batches, topk) -> dict:
    eeg_encoder.eval()
    total_loss = 0.0
    seen = 0
    eeg_embeddings = []
    texts: list[str] = []

    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        eeg = batch["eeg"].to(device, non_blocking=True)
        text_emb = text_encoder.encode_unique(batch["text"])
        eeg_emb = eeg_encoder(eeg)

        total_loss += info_nce(eeg_emb, text_emb, temperature).item() * len(batch["text"])
        seen += len(batch["text"])
        eeg_embeddings.append(eeg_emb.cpu())
        texts.extend(batch["text"])

    eeg_all = torch.cat(eeg_embeddings, dim=0)              # (N, D)
    text_all = text_encoder.encode_unique(texts).cpu()      # (N, D)，同序
    similarity = eeg_all @ text_all.t()                     # (N, N)
    targets = torch.arange(len(texts))

    k = min(topk, similarity.size(1))
    top1 = similarity.argmax(dim=1).eq(targets).float().mean().item()
    topk_acc = (
        similarity.topk(k, dim=1).indices
        .eq(targets[:, None])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )
    return {
        "loss": total_loss / seen,
        "top1": top1,
        "topk": topk_acc,
        "k": k,
        "cos_pos": similarity.diag().mean().item(),
        "n": len(texts),
    }


# ============================================================
# Main
# ============================================================

def base_dataset(loader):
    """从 DataLoader(Subset(ChiscoEEGDataset)) 里取回原始 dataset。"""
    dataset = loader.dataset
    return getattr(dataset, "dataset", dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Chisco-1.0 EEG→Text 检索模型")
    parser.add_argument("--subject", required=True, help="被试编号，如 01 或 sub-01")
    parser.add_argument("--task", required=True, choices=["read", "imagine"], help="任务类型")
    parser.add_argument("--day", type=int, default=1, help="day 编号，从 1 开始")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度")
    parser.add_argument("--drop-prob", type=float, default=0.5, help="TSConv dropout")
    parser.add_argument("--embedding-dim", type=int, default=768, help="embedding 维度")
    parser.add_argument("--topk", type=int, default=5, help="检索 top-k 的 k")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument("--text-model", default="sentence-transformers/LaBSE", help="文本编码模型名")
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="每轮最多跑多少个 batch（0 表示不限制，用于冒烟测试）",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---------------- 数据 ----------------
    train_loader, val_loader = build_dataloaders(
        subject=args.subject,
        task=args.task,
        day=args.day,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=True,
        # 只有 CUDA 能真正锁页：GPU 上开着让 .to(device, non_blocking=True)
        # 生效；CPU 上开着不会报错，但只会刷一条 UserWarning 且无收益。
        pin_memory=(device.type == "cuda"),
    )
    dataset = base_dataset(train_loader)
    n_chans, n_times = dataset.n_chans, dataset.n_times

    # ---------------- 模型 ----------------
    eeg_encoder = TSConvFixedEEGEncoder(
        n_chans=n_chans,
        n_times=n_times,
        drop_prob=args.drop_prob,
        embedding_dim=args.embedding_dim,
    ).to(device)

    text_encoder = LaBSETextEncoder(
        model_name=args.text_model,
    ).to(device)
    
    if text_encoder.embedding_dim != args.embedding_dim:
        raise ValueError(
            f"embedding 维度不一致: 文本 {text_encoder.embedding_dim} "
            f"vs 模型 {args.embedding_dim}"
        )

    optimizer = torch.optim.AdamW(
        eeg_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    n_params = sum(p.numel() for p in eeg_encoder.parameters() if p.requires_grad)
    print(
        f"{dataset.subject} / {dataset.task} / day-{dataset.day:02d}\n"
        f"  device        : {device}\n"
        f"  EEG           : ({n_chans}, {n_times})\n"
        f"  train / val   : {len(train_loader.dataset)} / {len(val_loader.dataset)}\n"
        f"  参数 / 温度   : {n_params:,} / {args.temperature}\n"
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # 文件名带运行时间戳，避免多次运行互相覆盖。
    run_id = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = (
        CHECKPOINT_DIR
        / f"{dataset.subject}_task-{dataset.task}_day-{dataset.day:02d}_{run_id}_best.pt"
    )

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(
            eeg_encoder, train_loader, text_encoder, optimizer,
            device, args.temperature, args.max_batches,
        )
        val_metrics = evaluate(
            eeg_encoder, val_loader, text_encoder, device,
            args.temperature, args.max_batches, args.topk,
        )
        dt = time.perf_counter() - t0

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f} top1 {train_metrics['top1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"top1 {val_metrics['top1']:.3f} "
            f"top{val_metrics['k']} {val_metrics['topk']:.3f} "
            f"cos+ {val_metrics['cos_pos']:.3f} | "
            f"{dt:.1f}s"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            torch.save(
                {
                    "model_state": eeg_encoder.state_dict(),
                    "args": vars(args),
                    "n_chans": n_chans,
                    "n_times": n_times,
                    "epoch": epoch,
                    "val_top1": val_metrics["top1"],
                    "val_topk": val_metrics["topk"],
                },
                ckpt_path,
            )
            print(f"  ↑ best top1 {best_top1:.3f}，已保存 {ckpt_path.name}")

    print(f"\n完成。最佳 val top1 = {best_top1:.3f}\n检查点: {ckpt_path}")


if __name__ == "__main__":
    main()
