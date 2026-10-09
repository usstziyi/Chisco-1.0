"""训练 Chisco-1.0 的 EEG→Text 检索模型（对比学习）。

组合的三个文件:
    encoder/eeg_encoder_tsconv_fixed.py  → TSConvFixedEEGEncoder
    encoder/text_encoder_labse.py        → LaBSETextEncoder（冻结，带文本缓存）
    scripts/dataloader.py                → 多 day 合并 + 跨天去重后划分 train / val

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
    重复句子已在两层去掉：
        prepare_dataset.py 保存 npz 时去掉同一天内重复的句子；
        dataloader.build_dataset 去掉跨 day 重复的句子。
    因此无论传单个还是多个 --days，每个句子在数据集里只出现一次，
    in-batch 对比不存在「重复句子 = 假负样本」问题，val 的 top-k 也
    按「第 i 条 EEG 对应第 i 条文本」成立。
    LaBSE 冻结且按文本做内存缓存，多 epoch 下每条文本只编码一次。

用法:
    uv run python scripts/train.py --subject 01 --task imagine --days 1 2 3 4 5
    uv run python scripts/train.py --subject 01 --task imagine --days 1 2 3 4 5 --epochs 50
    uv run python scripts/train.py --subject 01 --task read --days 1 2 3 4 5 --epochs 50

快速冒烟（只跑少量 batch）:
    uv run python scripts/train.py --subject 01 --task imagine --days 1 2 3 4 5 --max-batches 3
"""

import argparse
import csv
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

from dataloader import build_dataloaders, normalize_days, normalize_subject
from encoder import LaBSETextEncoder, TSConvFixedEEGEncoder


OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 每轮 epoch 追加写入的日志字段
EPOCH_LOG_FIELDS = [
    "epoch", "train_loss", "train_top1",
    "val_loss", "val_top1", "val_topk", "val_k", "val_cos_pos",
    "is_best", "best_top1", "seconds",
]

# 本次运行最终 val 结果的汇总字段
FINAL_VAL_FIELDS = [
    "subject", "task", "days", "run_id", "device",
    "n_chans", "n_times", "epochs", "best_epoch",
    "best_val_loss", "best_val_top1", "best_val_topk", "best_val_k", "best_val_cos_pos",
    "last_val_loss", "last_val_top1", "last_val_topk", "last_val_cos_pos",
    "checkpoint",
]


def append_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_row(path: Path, fields: list, row: dict) -> None:
    """把一行写入 CSV（覆盖已有内容）。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


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
    total_steps = min(len(loader), max_batches) if max_batches else len(loader)
    for step, batch in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        text_emb = text_encoder.encode_unique(batch["text"])
        eeg_emb = eeg_encoder(batch["eeg"].to(device, non_blocking=True))
        loss = info_nce(eeg_emb, text_emb, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = len(batch["text"])
        top1 = in_batch_top1(eeg_emb.detach(), text_emb)

        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        seen += batch_size

        print(
            f"  step {step + 1:4d}/{total_steps} | "
            f"loss {loss.item():.4f} | top1 {top1:.3f} | bs {batch_size}",
            flush=True,
        )
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

def probe_eeg_shape(loader) -> tuple[int, int]:
    """取一条样本的 EEG shape，得到 (n_chans, n_times)。

    不能直接用 loader.dataset.n_chans：多 day 或去重后拿到的是
    ConcatDataset / Subset，都不带 n_chans / n_times 属性。
    """
    return tuple(loader.dataset[0]["eeg"].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Chisco-1.0 EEG→Text 检索模型")
    parser.add_argument("--subject", required=True, help="被试编号，如 01 或 sub-01")
    parser.add_argument("--task", required=True, choices=["read", "imagine"], help="任务类型")
    parser.add_argument("--days", type=int, nargs="+", required=True, help="day 编号，可传多个，如 --days 1 2 3")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度")
    parser.add_argument("--drop-prob", type=float, default=0.5, help="TSConv dropout")
    parser.add_argument("--topk", type=int, default=5, help="检索 top-k 的 k")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    parser.add_argument("--text-model", default="sentence-transformers/LaBSE", help="文本编码模型名")
    parser.add_argument("--max-batches", type=int, default=0, help="每轮最多跑多少个 batch（0 表示不限制，用于冒烟测试）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---------------- 数据 ----------------
    subject = normalize_subject(args.subject)
    days = normalize_days(args.days)
    days_tag = "-".join(f"{d:02d}" for d in days)

    train_loader, val_loader = build_dataloaders(
        subject=subject,
        task=args.task,
        days=days,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=True,
        # 只有 CUDA 能真正锁页：GPU 上开着让 .to(device, non_blocking=True)
        # 生效；CPU 上开着不会报错，但只会刷一条 UserWarning 且无收益。
        pin_memory=(device.type == "cuda"),
    )
    n_chans, n_times = probe_eeg_shape(train_loader)

    # ---------------- 模型 ----------------
    text_encoder = LaBSETextEncoder(
        model_name=args.text_model,
    ).to(device)

    eeg_encoder = TSConvFixedEEGEncoder(
        n_chans=n_chans,
        n_times=n_times,
        k=40,
        m1=50,
        m2=100,
        s=20,
        projection_hidden_dim=512,
        embedding_dim=text_encoder.embedding_dim,
        drop_prob=args.drop_prob
    ).to(device)


    optimizer = torch.optim.AdamW(
        eeg_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    n_params = sum(p.numel() for p in eeg_encoder.parameters() if p.requires_grad)
    print(
        f"{subject} / {args.task} / days={days_tag}\n"
        f"  device        : {device}\n"
        f"  EEG           : ({n_chans}, {n_times})\n"
        f"  train / val   : {len(train_loader.dataset)} / {len(val_loader.dataset)}\n"
        f"  参数 / 温度   : {n_params:,} / {args.temperature}\n"
    )
    # --------------- 输出目录 ---------------
    # 每次运行新建一个独立文件夹：outputs/<被试>_task-<任务>_days-<day列表>_<时间戳>/
    # 文件夹名带被试/任务/days 和时间戳，便于区分多次运行。
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{subject}_task-{args.task}_days-{days_tag}_{run_id}"
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"          # 最佳权重
    epoch_log_path = run_dir / "epoch_log.csv"  # 每个 epoch 追加一行
    final_val_path = run_dir / "final_val.csv"  # 本次运行最终 val 结果
    print(f"  输出目录      : {run_dir}\n")

    # ---------------- 训练/验证 ----------------
    # 记录最佳 val top1 的 epoch、top1、topk、cos+
    best_top1 = -1.0
    best_epoch = 0
    best_val_metrics = None
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

        is_best = val_metrics["top1"] > best_top1
        if is_best:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
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

        # 把本轮日志追加到本次运行的 CSV
        append_csv_row(epoch_log_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_top1": f"{train_metrics['top1']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_top1": f"{val_metrics['top1']:.6f}",
            "val_topk": f"{val_metrics['topk']:.6f}",
            "val_k": val_metrics["k"],
            "val_cos_pos": f"{val_metrics['cos_pos']:.6f}",
            "is_best": int(is_best),
            "best_top1": f"{best_top1:.6f}",
            "seconds": f"{dt:.3f}",
        })

        if is_best:
            print(f"  ↑ best top1 {best_top1:.3f}，已保存 {ckpt_path.name}")

    # 本次运行最终 val 结果（含最佳与最后一轮）
    write_csv_row(final_val_path, FINAL_VAL_FIELDS, {
        "subject": subject,
        "task": args.task,
        "days": days_tag,
        "run_id": run_id,
        "device": str(device),
        "n_chans": n_chans,
        "n_times": n_times,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": f"{best_val_metrics['loss']:.6f}",
        "best_val_top1": f"{best_val_metrics['top1']:.6f}",
        "best_val_topk": f"{best_val_metrics['topk']:.6f}",
        "best_val_k": best_val_metrics["k"],
        "best_val_cos_pos": f"{best_val_metrics['cos_pos']:.6f}",
        "last_val_loss": f"{val_metrics['loss']:.6f}",
        "last_val_top1": f"{val_metrics['top1']:.6f}",
        "last_val_topk": f"{val_metrics['topk']:.6f}",
        "last_val_cos_pos": f"{val_metrics['cos_pos']:.6f}",
        "checkpoint": ckpt_path.name,
    })

    print(
        f"\n完成。最佳 val top1 = {best_top1:.3f} (epoch {best_epoch})\n"
        f"运行目录  : {run_dir}\n"
        f"检查点    : {ckpt_path}\n"
        f"逐轮日志  : {epoch_log_path.name}\n"
        f"最终结果  : {final_val_path.name}"
    )


if __name__ == "__main__":
    main()
