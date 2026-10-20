"""classification：单被试单 task 的 EEG 句子语义类别分类实验。

与 ``baseline/train.py`` 的对比学习不同，这里把任务改成**多分类**：输入整句 EEG
``(122, T)``，预测该句的语义类别 ``type_number``（0 ~ 38）。

模型
----
``EEGTypeClassifier`` = baseline 的 ``ConvTimeAttentionEncoder``（TSConv 前端 +
时间注意力池化）+ Dropout + 线性分类头，见 ``classification/classifier.py``。

目标函数
--------
带类别权重与标签平滑的交叉熵：

    loss = CrossEntropy(logits, label,
                        weight=class_weight,
                        label_smoothing=label_smoothing)

- ``class_weight``：在**训练集**上按反频率统计 ``N / (C * n_c)``。CSV 的类别分布很
  不均（单被试单 task 下最多 297 句、最少 40 句），不加权时模型会偏向头部类别，
  macro-F1 很差；``--class-weight none`` 可关掉。K 折模式下每折按自己的训练集重算。
- ``label_smoothing`` 默认 0.1：类别只有几十个样本，硬标签容易过拟合，平滑一点更稳。

评测与模型选择
--------------
两种模式，由 ``--n-folds`` 切换：

- ``--n-folds 1``（默认）：一次分层留出划分，**每轮都验证**，按 ``--select-by``
  （默认 ``macro_f1``）保存 best checkpoint。适合快速迭代、调超参。
- ``--n-folds K``（K >= 2）：K 折分层交叉验证，每折各自训练一个模型，**只在最后
  一轮验证一次**，不做 epoch 选择——否则就等于用验证折挑模型，报出来的指标偏乐观。
  最终输出 K 折的 ``mean ± std``，并把各折 held-out 的预测拼起来做 per-class 分析。

``val_loss`` 越小越好；``val_accuracy`` / ``val_macro_f1`` 越大越好。

数据划分
--------
- 单次留出（``--n-folds 1``）：按 ``type_number`` **分层**切分 train / val，每类内
  独立抽 ``--val-ratio`` 的句子；样本数 >= 2 的类别保证两侧都有。
- K 折（``--n-folds K``）：按 ``type_number`` 分层，每类打乱后轮流分配到 K 折
  （类内每折条数相差不超过 1），第 k 折当验证、其余 K-1 折合并训练。每个样本
  恰好被验证一次。

CSV 已按句子去重，所以不存在同句跨集泄漏。

数据加载（本机特有的约束）
--------------------------
``ChiscoEEGTypeDataset`` 每访问一个 run 就要完整 unpickle 一个约 465 MB 的 pkl，
再用 LRU 缓存（默认只留 2 个 run）；45 个 run 解包态合计约 21 GB，本机内存装不下。
所以这里沿用 baseline 的 ``RunBatchSampler``：按 run 分组后再切 batch，每个 epoch
只打乱 run 顺序和 run 内样本顺序，使每个 run 每轮只被加载一次。``num_workers``
必须为 0，否则每个 worker 都有自己的缓存副本，内存会成倍膨胀。

输出
----
``outputs/classification/<timestamp>/``：

- ``args.json``：本次命令行参数
- 单次留出：``<subject>_<task>_best.pt`` / ``_train_epochs.csv`` / ``_train_steps.csv``
- K 折：``<subject>_<task>_fold{k}.pt`` / ``_fold{k}_train_steps.csv`` /
  ``_cv_summary.csv``（每折一行指标）

用法::

    uv run python classification/train.py
    uv run python classification/train.py --subject sub-01 --task imagine
    uv run python classification/train.py --task reading --epochs 20
    uv run python classification/train.py --n-folds 5
    uv run python classification/train.py --subject sub-02 --batch-size 32 --max-runs 5   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

# 支持 `python classification/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.classifier import EEGTypeClassifier
from classification.dataset import ChiscoEEGTypeDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "classification"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss",
    "val_loss", "val_accuracy", "val_macro_f1", "n_val",
    "is_best", "label_smoothing", "class_weight",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = ["epoch", "step", "loss", "lr", "batch_size"]

# K 折模式下每折一行
CV_SUMMARY_FIELDS = [
    "fold", "n_train", "n_test", "train_loss", "val_loss",
    "val_accuracy", "val_macro_f1",
]


class RunBatchSampler(Sampler[list[int]]):
    """按 run 分组的 batch 采样器，见模块文档「数据加载」一节。

    产出的每个 batch 是一串**全局**样本下标（相对底层 dataset），所以在
    ``DataLoader`` 里必须把完整的 dataset 传进去，不能再套 ``Subset``。

    Args:
        dataset: ``ChiscoEEGTypeDataset``，只用来读 ``rows`` 里的 run 编号。
        indices: 本次要遍历的全局样本下标（例如 train / val 各自的那一半）。
        batch_size: 每个 batch 的样本数。
        shuffle: 是否打乱 run 顺序与 run 内样本顺序；验证集传 False。
        drop_last: 是否丢掉最后一个不满的 batch。
        seed: 打乱用的种子；实际种子是 ``seed + epoch``，保证每个 epoch 不同。
    """

    def __init__(
        self,
        dataset: ChiscoEEGTypeDataset,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # run -> 该集合内属于这个 run 的样本下标
        self.by_run: dict[int, list[int]] = {}
        for index in indices:
            run = int(dataset.rows[index]["run"])
            self.by_run.setdefault(run, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        """由训练循环每轮调用，让每个 epoch 的打乱结果不同。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        runs = sorted(self.by_run)
        if self.shuffle:
            rng.shuffle(runs)

        batches: list[list[int]] = []
        pending: list[int] = []
        for run in runs:
            indices = list(self.by_run[run])
            if self.shuffle:
                rng.shuffle(indices)
            pending.extend(indices)
            # 攒够一个 batch 就切出去。只在 run 边界处，一个 batch 才会横跨相邻的
            # 若干个 run；产出的下标流里每个 run 仍是连续的一块，不会回头访问
            while len(pending) >= self.batch_size:
                batches.append(pending[: self.batch_size])
                pending = pending[self.batch_size :]

        if pending and not self.drop_last:
            batches.append(pending)
        return iter(batches)

    def __len__(self) -> int:
        total = sum(len(v) for v in self.by_run.values())
        if self.drop_last:
            return total // self.batch_size
        return -(-total // self.batch_size)


def append_csv_row(path: Path, fields: list[str], row: dict) -> None:
    """把一行追加到 CSV；文件不存在时先写表头。"""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def dump_args_json(args, out_dir: Path, filename: str = "args.json") -> Path:
    """把本次运行的所有命令行参数（key/value）以 JSON 写到时间戳输出目录。"""
    path = Path(out_dir) / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_stratified(
    dataset: ChiscoEEGTypeDataset,
    indices: Sequence[int],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], dict]:
    """按 ``type_number`` 分层切分 train / val。

    每个类别内部独立打乱后，按比例抽取验证样本；样本数 >= 2 的类别保证至少
    1 条进验证集、至少 1 条留在训练集；只有 1 条的类别整个留在训练集。
    """
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio 需在 [0, 1) 内，收到 {val_ratio}")

    rng = random.Random(seed)
    by_type: dict[int, list[int]] = {}
    for index in indices:
        by_type.setdefault(int(dataset.rows[index]["type_number"]), []).append(index)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for type_number in sorted(by_type):
        group = list(by_type[type_number])
        rng.shuffle(group)
        if len(group) < 2:
            n_val = 0
        else:
            n_val = int(round(len(group) * val_ratio))
            n_val = min(max(n_val, 1), len(group) - 1)
        val_idx.extend(group[:n_val])
        train_idx.extend(group[n_val:])

    if not train_idx or not val_idx:
        raise ValueError(
            f"分层切分后 train={len(train_idx)} / val={len(val_idx)}，样本不足"
        )

    train_types = {int(dataset.rows[i]["type_number"]) for i in train_idx}
    val_types = {int(dataset.rows[i]["type_number"]) for i in val_idx}
    info = {
        "n_classes": len(by_type),
        "train_classes": len(train_types),
        "val_classes": len(val_types),
    }
    return train_idx, val_idx, info


def split_kfold(
    dataset: ChiscoEEGTypeDataset,
    indices: Sequence[int],
    n_folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """按 ``type_number`` 分层做 K 折切分，返回 ``[(train_idx, val_idx), ...]``。

    每个类别内部独立打乱后轮流分配到各折，因此类内每折的条数相差不超过 1，
    每个样本恰好被验证一次。
    """
    if n_folds < 2:
        raise ValueError(f"n_folds 必须 >= 2，收到 {n_folds}")

    rng = random.Random(seed)
    by_type: dict[int, list[int]] = {}
    for index in indices:
        by_type.setdefault(int(dataset.rows[index]["type_number"]), []).append(index)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for type_number in sorted(by_type):
        group = list(by_type[type_number])
        rng.shuffle(group)
        for offset, index in enumerate(group):
            folds[offset % n_folds].append(index)

    splits: list[tuple[list[int], list[int]]] = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = [index for j in range(n_folds) if j != k for index in folds[j]]
        if not train_idx or not val_idx:
            raise ValueError(f"第 {k + 1} 折为空，请减少 --n-folds 或检查样本量")
        splits.append((train_idx, val_idx))
    return splits


def compute_class_weights(
    dataset: ChiscoEEGTypeDataset,
    indices: Sequence[int],
    n_classes: int,
) -> torch.Tensor:
    """按训练集反频率计算交叉熵的类别权重 ``N / (C * n_c)``。

    训练集里没出现的类别权重置 1.0（它不会产生正样本，权重对损失无贡献）。
    """
    counts = Counter(dataset.label_of(i) for i in indices)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("训练集为空，无法计算类别权重")
    weights = [
        total / (n_classes * counts[c]) if counts.get(c, 0) > 0 else 1.0
        for c in range(n_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def report_loss_config(
    args,
    dataset: ChiscoEEGTypeDataset,
    train_idx: Sequence[int],
    n_classes: int,
    note: str = "",
) -> None:
    """打印目标函数配置（类别权重按给定训练集统计）。"""
    if args.class_weight != "inverse":
        print(f"[Loss] CrossEntropy(label_smoothing={args.label_smoothing})，无类别权重")
        return
    weights = compute_class_weights(dataset, train_idx, n_classes)
    print(
        f"[Loss] CrossEntropy(label_smoothing={args.label_smoothing}) "
        f"+ 反频率类别权重 min/max={weights.min():.4f}/{weights.max():.4f}{note}"
    )


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """返回 ``(accuracy, macro_f1)``。

    macro-F1 只在验证集实际出现的类别上取平均（没有正样本的类别不参与）。
    """
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())

    f1_scores: list[float] = []
    for c in labels.unique().tolist():
        true_mask = labels == c
        pred_mask = predictions == c
        tp = float((true_mask & pred_mask).sum())
        fp = float((~true_mask & pred_mask).sum())
        fn = float((true_mask & ~pred_mask).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        )
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return accuracy, macro_f1


def build_loader(
    dataset: ChiscoEEGTypeDataset,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
):
    """建一个 run-sequential 的 DataLoader，同时返回它用的采样器。"""
    sampler = RunBatchSampler(
        dataset,
        indices,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # 训练时丢掉最后一个不满 batch，验证时保留
        seed=seed,
    )
    # 注意：batch_sampler 产出的下标是相对完整 dataset 的，所以这里不能传 Subset；
    # 也不能再传 batch_size / shuffle / drop_last（与 batch_sampler 互斥）。
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    return loader, sampler


def train_one_epoch(
    model,
    loader,
    sampler,
    criterion,
    optimizer,
    device,
    epoch,
    step_log_path=None,
):
    """跑一个 epoch，返回平均 loss。"""
    model.train()
    sampler.set_epoch(epoch)
    total_loss, total_n = 0.0, 0
    lr = optimizer.param_groups[0]["lr"]
    for step, (eeg, labels) in enumerate(loader, start=1):
        eeg = eeg.to(device)          # (B, C, T)
        labels = labels.to(device)    # (B,)

        logits = model(eeg)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = labels.size(0)
        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr:.6f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_n += batch
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model, loader, sampler, device) -> tuple[torch.Tensor, torch.Tensor]:
    """在整个 loader 上收集 logits 与标签。"""
    model.eval()
    sampler.set_epoch(0)  # 验证集不打乱，固定顺序便于复现

    logits_parts, labels_parts = [], []
    for eeg, labels in loader:
        logits_parts.append(model(eeg.to(device)))
        labels_parts.append(labels.to(device))
    return torch.cat(logits_parts, dim=0), torch.cat(labels_parts, dim=0)


def train_fold(
    args,
    dataset: ChiscoEEGTypeDataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    n_chans: int,
    n_samples: int,
    n_classes: int,
    device,
    out_dir: Path,
    tag: str,
    seed: int,
    eval_each_epoch: bool,
) -> dict:
    """在一个 ``(train, val)`` 划分上训练并返回最终验证结果。

    ``eval_each_epoch=True``：每轮验证，按 ``--select-by`` 保存 best checkpoint，
    训练结束后再载入 best 重新预测（单次留出模式）。
    ``eval_each_epoch=False``：只在最后一轮验证一次并保存该轮 checkpoint，held-out
    那一份不参与任何模型选择（K 折模式）。
    """
    set_seed(seed)

    train_loader, train_sampler = build_loader(
        dataset, train_idx, args.batch_size, True, args.seed, args.num_workers
    )
    val_loader, val_sampler = build_loader(
        dataset, val_idx, args.batch_size, False, args.seed, args.num_workers
    )

    model = EEGTypeClassifier(
        n_samples=n_samples,
        n_chans=n_chans,
        n_classes=n_classes,
        embedding_dim=args.embedding_dim,
        head_dropout=args.head_dropout,
        logit_scale=args.logit_scale,
    ).to(device)

    weights = None
    if args.class_weight == "inverse":
        weights = compute_class_weights(dataset, train_idx, n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    ckpt_path = out_dir / f"{tag}.pt"
    steps_path = out_dir / f"{tag}_train_steps.csv"
    epochs_path = out_dir / f"{tag}_train_epochs.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_epoch = 0
    best_metrics = None
    logits = labels = None
    train_loss = 0.0

    if eval_each_epoch:
        header = (
            f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'val_loss':>9} "
            f"{'val_acc':>8} {'val_f1':>8}"
        )
        print(f"\n{header}  (best by {args.select_by})")
        print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, train_sampler, criterion, optimizer, device,
            epoch=epoch, step_log_path=steps_path,
        )
        if scheduler is not None:
            scheduler.step()

        # K 折模式：中间轮次不做评估，省掉整轮 held-out 的 pkl 解码
        if not eval_each_epoch and epoch < args.epochs:
            print(f"  epoch {epoch:>3}/{args.epochs} train_loss={train_loss:.4f}")
            continue

        logits, labels = predict(model, val_loader, val_sampler, device)
        val_loss = float(criterion(logits, labels))
        accuracy, macro_f1 = classification_metrics(logits, labels)
        metrics = {
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_val": int(labels.numel()),
        }

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(metrics)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": n_chans,
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                    "classes": dataset.classes,
                    "metrics": metrics,
                },
                ckpt_path,
            )

        if eval_each_epoch:
            flag = " *" if improved else ""
            print(
                f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} "
                f"{val_loss:>9.4f} {accuracy:>8.4f} {macro_f1:>8.4f}{flag}"
            )
            # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
            append_csv_row(epochs_path, EPOCH_LOG_FIELDS, {
                "epoch": epoch,
                "lr": f"{epoch_lr:.6f}",
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{accuracy:.6f}",
                "val_macro_f1": f"{macro_f1:.6f}",
                "n_val": metrics["n_val"],
                "is_best": int(improved),
                "label_smoothing": args.label_smoothing,
                "class_weight": args.class_weight,
            })

    # 单次留出模式：per-class 报告要用 best checkpoint 的预测，重新载入一次
    if eval_each_epoch and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logits, labels = predict(model, val_loader, val_sampler, device)

    return {
        "train_loss": train_loss,
        "logits": logits,
        "labels": labels,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "ckpt_path": ckpt_path,
    }


def report_class_distribution(dataset: ChiscoEEGTypeDataset, indices: Sequence[int]) -> None:
    """按 type_number 升序打印「类别: 个数」，与标签编号顺序一致。"""
    counts = Counter(int(dataset.rows[i]["type_number"]) for i in indices)
    print(f"类别分布（{len(indices)} 句，{len(counts)} 类）：")
    for type_number in sorted(counts):
        name = dataset.type_names.get(type_number, "")
        print(f"  {type_number:02d} {name}: {counts[type_number]}")


def report_per_class(
    dataset: ChiscoEEGTypeDataset,
    logits: torch.Tensor,
    labels: torch.Tensor,
    split_name: str = "验证集",
) -> None:
    """打印每类召回率（只列该集合里出现的类别）。"""
    predictions = logits.argmax(dim=1)
    print(f"\n[Per-class] {split_name}每类召回率：")
    for c in labels.unique().tolist():
        type_number = dataset.classes[c]
        name = dataset.type_names.get(type_number, "")
        mask = labels == c
        support = int(mask.sum())
        recall = float((predictions[mask] == c).float().mean())
        print(f"  {type_number:02d} {name:<12} support={support:>3} recall={recall:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="classification：单被试单 task 的 EEG 语义类别分类。")
    parser.add_argument("--subject", default="sub-01", help="被试编号，如 sub-01（默认 sub-01）")
    parser.add_argument("--task", default="imagine", choices=["reading", "imagine"], help="reading / imagine（默认 imagine）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，按类别分层抽样（默认 0.2，仅 --n-folds 1 时生效）")
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="K 折分层交叉验证：1 = 单次留出划分（默认）；>=2 时每折只在最后一轮验证一次、不做 epoch 选择",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch 大小（默认 64）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"], help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定")
    parser.add_argument("--select-by",default="macro_f1",choices=["val_loss", "accuracy", "macro_f1"],help="用哪个验证指标选 best checkpoint（默认 macro_f1，越大越好；仅 --n-folds 1 时生效）")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="交叉熵标签平滑系数（默认 0.1）")
    parser.add_argument("--class-weight",default="inverse",choices=["none", "inverse"],help="类别权重：inverse 按训练集反频率加权（默认），none 不加权")
    parser.add_argument("--embedding-dim", type=int, default=128, help="编码器输出维度（默认 128）")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="分类头前的 dropout（默认 0.3）")
    parser.add_argument("--logit-scale", type=float, default=16.0, help="余弦分类器的固定缩放系数（默认 16）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--max-runs", type=int, default=0, help="只用 run 编号最小的前 N 个 run（0 = 全部），便于快速冒烟")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数，必须为 0（见模块文档）")
    parser.add_argument("--device", default=None, help="cpu / cuda，默认自动选择")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_workers != 0:
        raise ValueError(
            "num-workers 必须为 0：每个 worker 都会自带一份 pkl 缓存，"
            "既会让 LRU 命中率崩掉，也会成倍占用内存（单 run 解包后约 465 MB）"
        )
    if args.n_folds < 1:
        raise ValueError("--n-folds 必须 >= 1")
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] device={device} subject={args.subject} task={args.task} "
          f"batch={args.batch_size} epochs={args.epochs} lr={args.lr} "
          f"wd={args.weight_decay} schedule={args.lr_schedule} "
          f"select={args.select_by} label_smoothing={args.label_smoothing} "
          f"class_weight={args.class_weight} n_folds={args.n_folds} seed={args.seed}")

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据 ----------------
    dataset = ChiscoEEGTypeDataset(args.subject, args.task)
    first_eeg, _ = dataset[0]  # 触发一次读取，确定通道数 / 时间长度
    n_chans, n_samples = first_eeg.shape
    n_classes = len(dataset.classes)

    indices = list(range(len(dataset)))
    if args.max_runs > 0:
        keep_runs = sorted({int(row["run"]) for row in dataset.rows})[: args.max_runs]
        keep = set(keep_runs)
        indices = [i for i in indices if int(dataset.rows[i]["run"]) in keep]

    # ---------------- 模型信息 ----------------
    # 只用来打印结构信息；train_fold 里会按 fold 自己的种子重新建模型
    probe = EEGTypeClassifier(
        n_samples=n_samples,
        n_chans=n_chans,
        n_classes=n_classes,
        embedding_dim=args.embedding_dim,
        head_dropout=args.head_dropout,
        logit_scale=args.logit_scale,
    )
    n_params = sum(p.numel() for p in probe.parameters())
    print(
        f"[Model] {type(probe).__name__} 参数量={n_params:,} "
        f"d_model={probe.encoder.d_model} k={probe.encoder.k} T2={probe.n_time}"
    )
    del probe

    tag = f"{args.subject}_{args.task}"
    data_header = (
        f"[Data] {args.subject} {args.task} 样本 {len(dataset)}"
        f"（使用 {len(indices)} 条 / {len({int(dataset.rows[i]['run']) for i in indices})} 个 run"
        f"，无效 type_number 丢弃 {dataset.n_skipped} 条）"
        f"；每个 run 只加载一次；n_chans={n_chans} T={n_samples}"
    )

    # ---------------- 单次留出：每轮验证 + 选 best ----------------
    if args.n_folds == 1:
        train_idx, val_idx, info = split_stratified(
            dataset, indices, args.val_ratio, args.seed
        )
        print(
            f"{data_header}\n"
            f"       类别 {n_classes} 个 -> train {info['train_classes']} 类 "
            f"({len(train_idx)} 条) / val {info['val_classes']} 类 ({len(val_idx)} 条)"
        )
        report_class_distribution(dataset, train_idx)
        report_loss_config(args, dataset, train_idx, n_classes)

        result = train_fold(
            args, dataset, train_idx, val_idx, n_chans, n_samples, n_classes,
            device, out_dir, tag, args.seed, eval_each_epoch=True,
        )
        metrics = result["metrics"]

        report_per_class(dataset, result["logits"], result["labels"])
        print(
            f"\n[Done] best epoch={result['best_epoch']} {args.select_by}="
            f"{metrics[args.select_by]:.4f} "
            f"(val_acc={metrics['accuracy']:.4f}, "
            f"val_macro_f1={metrics['macro_f1']:.4f}, "
            f"val_loss={metrics['val_loss']:.4f})\n"
            f"       随机基线 accuracy≈{1 / n_classes:.4f}（{n_classes} 类）\n"
            f"       checkpoint: {result['ckpt_path']}\n"
            f"       epochs:     {out_dir / f'{tag}_train_epochs.csv'}\n"
            f"       steps:      {out_dir / f'{tag}_train_steps.csv'}"
        )
        return

    # ---------------- K 折分层交叉验证 ----------------
    splits = split_kfold(dataset, indices, args.n_folds, args.seed)
    print(
        f"{data_header}\n"
        f"       {args.n_folds} 折分层交叉验证：每折只在最后一轮验证一次，不做 epoch 选择\n"
        f"       类别 {n_classes} 个；各折 train/val 条数："
        + " | ".join(f"F{k + 1} {len(tr)}/{len(va)}" for k, (tr, va) in enumerate(splits))
    )
    report_class_distribution(dataset, indices)
    report_loss_config(
        args, dataset, splits[0][0], n_classes, note="（各折按自己的训练集重算，这里显示 fold 1）"
    )

    summary_path = out_dir / f"{tag}_cv_summary.csv"
    fold_metrics: list[dict] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_tag = f"{tag}_fold{fold}"
        print(f"\n===== Fold {fold}/{args.n_folds}：train {len(train_idx)} 条 / val {len(val_idx)} 条 =====")
        result = train_fold(
            args, dataset, train_idx, val_idx, n_chans, n_samples, n_classes,
            device, out_dir, fold_tag, args.seed + fold, eval_each_epoch=False,
        )
        metrics = result["metrics"]
        fold_metrics.append(metrics)
        all_logits.append(result["logits"])
        all_labels.append(result["labels"])

        print(
            f"[Fold {fold}] train_loss={result['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )
        append_csv_row(summary_path, CV_SUMMARY_FIELDS, {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(val_idx),
            "train_loss": f"{result['train_loss']:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "val_accuracy": f"{metrics['accuracy']:.6f}",
            "val_macro_f1": f"{metrics['macro_f1']:.6f}",
        })

    print(f"\n[CV] {args.n_folds} 折结果（mean ± std，std 为折间标准差）：")
    for key, name in (
        ("val_loss", "val_loss"),
        ("accuracy", "val_acc"),
        ("macro_f1", "val_f1"),
    ):
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        per_fold = ", ".join(f"{v:.4f}" for v in values)
        print(f"  {name:<9} {values.mean():.4f} ± {values.std(ddof=1):.4f}   [{per_fold}]")

    pooled_logits = torch.cat(all_logits, dim=0)
    pooled_labels = torch.cat(all_labels, dim=0)
    pooled_acc, pooled_f1 = classification_metrics(pooled_logits, pooled_labels)
    print(
        f"\n[CV] 汇总 held-out（{pooled_labels.numel()} 条，每个样本恰好一次）："
        f"accuracy={pooled_acc:.4f} macro_f1={pooled_f1:.4f}"
    )
    report_per_class(dataset, pooled_logits, pooled_labels, split_name="held-out ")

    print(
        f"\n[Done] {args.n_folds}-fold CV 完成\n"
        f"       summary: {summary_path}\n"
        f"       随机基线 accuracy≈{1 / n_classes:.4f}（{n_classes} 类）"
    )


if __name__ == "__main__":
    main()
