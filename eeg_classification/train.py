"""eeg_classification 的训练循环：只编码 EEG，做句子语义类别（39 类）分类。

与 ``baseline/train.py`` 的对比
------------------------------
- **没有文本侧**：去掉 LaBSE，也就没有了"文本 embedding 查表"和对比损失；
- **标签是离散类别**：``type_number``（0~38），损失换成带类别权重 /
  标签平滑的 ``CrossEntropyLoss``；
- **指标换成分类指标**：accuracy、top-5 accuracy、macro-F1，以及最好那轮的混淆矩阵；
- 其余（数据访问、run-sequential 采样、按句子切分、日志格式）与 baseline 保持一致，
  便于实验之间横向比较。

数据
----
EEG 来自 ``datasets/ds005170-download/derivatives/preprocessed_pkl``，
标签来自 ``datasets/preprocessed/{subject}_{task}.csv`` 的 ``type_number`` 列。
一次只跑一个 ``(subject, task)``。

train / val 按 **句子** 切分（不是按样本）：本数据集同一句话可能在不同 run 里
重复出现，按样本随机切分会让同一句话的多个试次同时落在两侧，val 指标被泄漏污染。
按句子切分后，val 里的每句话都是训练时没见过的，但类别（39 类）两侧共有——
这正是分类任务想要的：测"能不能从没见过的句子的 EEG 里解出语义类别"。

数据加载（本机特有的约束）
--------------------------
``ChiscoEEGClassificationDataset`` 每访问一个 run 就要完整 unpickle 一个 465 MB 的
pkl，再用 LRU 缓存（默认只留 2 个 run）。若直接 ``shuffle=True``，缓存命中率会掉到
几个百分点，每个 batch 都要反复重载几十个文件，完全跑不动。所以这里用
``RunBatchSampler``：按 run 分组后再切 batch，每个 epoch 只打乱 run 顺序与 run 内
顺序，使每个 run 每轮只被加载一次；代价是 ``num_workers`` 必须为 0。

评测
----
- ``val_acc``：top-1 准确率；
- ``val_top5``：前 5 名命中率（39 类下比 top-1 敏感得多）；
- ``val_macro_f1``：按类别等权平均的 F1，分母固定为全部 39 类，
  能暴露"只学会大类"的问题（类别分布并不均衡）；
- ``val_loss``：带类别权重 / 标签平滑的交叉熵，用于看收敛。

best checkpoint 默认按 ``val_macro_f1`` 选（越大越好）；也可切到 ``val_acc`` /
``val_loss``。所有指标都按**全局 39 类**计算，跨被试可直接比较。

用法::

    uv run python eeg_classification/train.py
    uv run python eeg_classification/train.py --task imagine --epochs 20
    uv run python eeg_classification/train.py --subject sub-02 --max-runs 5 --epochs 1   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Sampler

# 支持 `python eeg_classification/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_classification.dataset import (
    ChiscoEEGClassificationDataset,
    load_class_names,
)
from eeg_classification.model import EEGClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "eeg_classification"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss", "train_acc",
    "val_loss", "val_acc", "val_top5", "val_macro_f1",
    "n_val", "n_classes_val", "is_best",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = ["epoch", "step", "loss", "acc", "lr", "batch_size"]


class RunBatchSampler(Sampler[list[int]]):
    """按 run 分组的 batch 采样器，副本自 ``baseline/train.py``。

    产出的每个 batch 是一串**全局**样本下标（相对底层 dataset），所以在
    ``DataLoader`` 里必须把完整的 dataset 传进去，不能再套 ``Subset``。

    Args:
        dataset: 分类 Dataset，只用来读 ``rows`` 里的 run 编号。
        indices: 本次要遍历的全局样本下标（train / val 各自的那一半）。
        batch_size: 每个 batch 的样本数。
        shuffle: 是否打乱 run 顺序与 run 内样本顺序；验证集传 False。
        drop_last: 是否丢掉最后一个不满的 batch。
        seed: 打乱用的种子；实际种子是 ``seed + epoch``，保证每个 epoch 不同。
    """

    def __init__(
        self,
        dataset: ChiscoEEGClassificationDataset,
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
            # 若干个 run；产出的下标流里每个 run 仍是连续的一块，不会回头访问。
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


def split_by_sentence(
    dataset: ChiscoEEGClassificationDataset,
    indices: Sequence[int],
    val_ratio: float,
    seed: int,
):
    """按句子切分 train / val，返回 ``(train_idx, val_idx, 统计信息)``。

    以句子为单位而非样本：同一句话可能有多条试次，若按样本随机切分，同一句话会
    同时出现在两侧，val 指标会被泄漏彻底污染。类别本身当然两侧都有，这是分类任务
    的目标，不算泄漏。
    """
    sentences = [dataset.rows[i]["sentence"] for i in indices]
    unique = sorted(set(sentences))
    if len(unique) < 2:
        raise ValueError(f"只有 {len(unique)} 个唯一句子，无法切分")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_sentences = set(shuffled[:n_val])

    train_idx = [i for i, s in zip(indices, sentences) if s not in val_sentences]
    val_idx = [i for i, s in zip(indices, sentences) if s in val_sentences]

    # 泄漏检查：按句子切分后，train / val 的句子集合必须完全不相交
    info = {
        "n_unique_train": len(unique) - n_val,
        "n_unique_val": n_val,
        "sentence_overlap": len(
            {dataset.rows[i]["sentence"] for i in train_idx} & val_sentences
        ),
    }
    return train_idx, val_idx, info


def build_class_weights(
    dataset: ChiscoEEGClassificationDataset,
    indices: Sequence[int],
    num_classes: int,
    mode: str,
) -> torch.Tensor | None:
    """按训练集类别频率算交叉熵的类别权重。

    ``inverse``：``w_c = sum(n) / (n_present * n_c)``，只对出现过的类算，未出现的类
    权重置 0（本来也不会成为训练目标）。所有出现类的权重均值约等于 1，不会整体改变
    损失量级。类别分布并不均衡，不加权的模型容易只学大类。
    """
    if mode == "none":
        return None

    counts = dataset.class_counts(indices)
    present = counts > 0
    if not present.any():
        return None

    weight = np.zeros(num_classes, dtype=np.float32)
    weight[present] = counts[present].sum() / (present.sum() * counts[present])
    return torch.tensor(weight, dtype=torch.float32)


def train_one_epoch(
    model, loader, sampler, criterion, optimizer, device, epoch, step_log_path=None,
):
    """跑一个 epoch，返回 ``(平均 loss, 训练准确率)``。

    传了 ``step_log_path`` 就把每个 step 追加到 CSV：第 0 列 epoch、第 1 列 step。
    """
    model.train()
    sampler.set_epoch(epoch)
    total_loss, total_correct, total_n = 0.0, 0, 0
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
        correct = logits.argmax(dim=1).eq(labels).sum().item()
        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "acc": f"{correct / batch:.6f}",
                "lr": f"{lr:.6f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_correct += correct
        total_n += batch
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, sampler, criterion, device, num_classes) -> tuple[dict, np.ndarray]:
    """在整个 val 集上算分类指标，返回 ``(metrics, 混淆矩阵)``。

    指标全部按**全局 num_classes 类**计算（即使某些类在 val 里没出现），
    保证不同被试 / 不同 task 之间可比。
    """
    model.eval()
    sampler.set_epoch(0)  # 验证集不打乱，固定顺序便于复现

    logits_parts, label_parts = [], []
    total_loss, total_n = 0.0, 0
    for eeg, labels in loader:
        eeg = eeg.to(device)
        labels = labels.to(device)
        logits = model(eeg)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        logits_parts.append(logits)
        label_parts.append(labels)

    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    pred = logits.argmax(dim=1)

    k = min(5, num_classes)
    topk = logits.topk(k, dim=1).indices
    label_np = labels.cpu().numpy()
    pred_np = pred.cpu().numpy()

    # 行 = 真实类别，列 = 预测类别
    confusion = (
        torch.bincount(
            labels * num_classes + pred, minlength=num_classes * num_classes
        )
        .reshape(num_classes, num_classes)
        .cpu()
        .numpy()
    )

    metrics = {
        "val_loss": total_loss / max(total_n, 1),
        "val_acc": pred.eq(labels).float().mean().item(),
        "val_top5": topk.eq(labels[:, None]).any(dim=1).float().mean().item(),
        # labels=range(num_classes)：缺席的类按 F1=0 计入，避免"少类表现"被掩盖
        "val_macro_f1": float(
            f1_score(
                label_np, pred_np, labels=list(range(num_classes)),
                average="macro", zero_division=0,
            )
        ),
        "n_val": int(labels.numel()),
        "n_classes_val": int(len(set(label_np.tolist()))),
    }
    return metrics, confusion


def save_confusion_csv(
    path: Path,
    confusion: np.ndarray,
    class_names: dict[int, str],
    num_classes: int,
) -> None:
    """把混淆矩阵写成 CSV：行 = 真实类别，列 = 预测类别（另有 support / recall 两列）。"""
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion), support,
        out=np.zeros(num_classes, dtype=np.float64), where=support > 0,
    )
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["true_id", "true_name", "support", "recall"]
            + [f"pred_{c}" for c in range(num_classes)]
        )
        for c in range(num_classes):
            writer.writerow(
                [c, class_names.get(c, ""), int(support[c]), f"{recall[c]:.4f}"]
                + confusion[c].tolist()
            )


def worst_classes(
    confusion: np.ndarray, class_names: dict[int, str], num_classes: int, top: int = 5
):
    """按召回率升序列出表现最差的类别（只看 val 里出现过的类）。"""
    support = confusion.sum(axis=1)
    shown = np.flatnonzero(support > 0)
    if shown.size == 0:
        return []
    recall = np.diag(confusion)[shown] / support[shown]
    order = np.argsort(recall)[:top]
    return [
        (int(shown[i]), class_names.get(int(shown[i]), ""), float(recall[i]), int(support[shown[i]]))
        for i in order
    ]


def build_loader(
    dataset: ChiscoEEGClassificationDataset,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="eeg_classification：只编码 EEG、预测句子 type_number 的 39 类分类。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试编号，如 sub-01（默认 sub-01）")
    parser.add_argument("--task", default="imagine", choices=["reading", "imagine"], help="reading / imagine（默认 imagine）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--batch-size", type=int, default=64, help="batch 大小（默认 64）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"], help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="交叉熵标签平滑系数（默认 0.0；类别细且样本少时可试 0.1）")
    parser.add_argument("--class-weight", default="none", choices=["none", "inverse"], help="交叉熵类别权重：none 不加权，inverse 按训练集频率倒数加权（默认 none）")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="分类头之前的 dropout（默认 0.3）")
    parser.add_argument("--select-by", default="val_macro_f1", choices=["val_macro_f1", "val_acc", "val_loss"], help="用哪个指标选 best checkpoint（默认 val_macro_f1，越大越好；val_loss 越小越好）")
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
            "既会让 LRU 命中率崩掉，也会成倍占用内存（单 run 解包后 465 MB）"
        )
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Config] device={device} subject={args.subject} task={args.task} "
          f"batch={args.batch_size} epochs={args.epochs} lr={args.lr} "
          f"wd={args.weight_decay} schedule={args.lr_schedule} "
          f"label_smoothing={args.label_smoothing} class_weight={args.class_weight} "
          f"head_dropout={args.head_dropout} select={args.select_by} seed={args.seed}")

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据 ----------------
    dataset = ChiscoEEGClassificationDataset(args.subject, args.task)
    first_eeg, _ = dataset[0]  # 触发一次读取，确定通道数 / 时间长度
    n_chans, n_samples = first_eeg.shape
    num_classes = dataset.num_classes

    # 类别 id -> 类别名，只用于报告（混淆矩阵、最差类别）
    class_names = load_class_names()
    class_names_path = out_dir / "class_names.json"
    with class_names_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(class_names.items())}, f, ensure_ascii=False, indent=2)

    indices = list(range(len(dataset)))
    if args.max_runs > 0:
        keep_runs = sorted({int(row["run"]) for row in dataset.rows})[: args.max_runs]
        keep = set(keep_runs)
        indices = [i for i in indices if int(dataset.rows[i]["run"]) in keep]

    train_idx, val_idx, info = split_by_sentence(
        dataset, indices, args.val_ratio, args.seed
    )
    counts = dataset.class_counts(indices)
    print(
        f"[Data] {args.subject} {args.task} 样本 {len(dataset)}"
        f"（使用 {len(indices)} 条 / {len({int(dataset.rows[i]['run']) for i in indices})} 个 run；"
        f"丢弃无标签 {dataset.n_dropped_unlabeled} 条）\n"
        f"       每个 run 只加载一次；n_chans={n_chans} T={n_samples} num_classes={num_classes}\n"
        f"       句子 {len(set(dataset.rows[i]['sentence'] for i in indices))} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(样本 {len(train_idx)} / {len(val_idx)})，句子重叠={info['sentence_overlap']}\n"
        f"       类别覆盖 {int((counts > 0).sum())}/{num_classes}，"
        f"每类样本 min/mean/max = "
        f"{counts[counts > 0].min()}/{counts[counts > 0].mean():.1f}/{counts.max()}"
    )

    train_loader, train_sampler = build_loader(
        dataset, train_idx, args.batch_size, True, args.seed, args.num_workers
    )
    val_loader, val_sampler = build_loader(
        dataset, val_idx, args.batch_size, False, args.seed, args.num_workers
    )

    # ---------------- 模型 ----------------
    model = EEGClassifier(
        n_samples=n_samples,
        n_classes=num_classes,
        n_chans=n_chans,
        head_dropout=args.head_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} k={model.k} T2={model.n_time} n_classes={num_classes}"
    )

    # ---------------- 损失 / 优化器 ----------------
    class_weights = build_class_weights(
        dataset, train_idx, num_classes, args.class_weight
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    # 评测用的 criterion 不带类别权重：val_loss 要在不同配置之间可比
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # 恒定学习率在指标走平后仍会继续拟合，余弦衰减到 0 有助于收敛
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # ---------------- 训练 ----------------
    tag = f"{args.subject}_{args.task}"
    ckpt_path = out_dir / f"{tag}_best.pt"
    train_epochs_path = out_dir / f"{tag}_train_epochs.csv"
    train_steps_path = out_dir / f"{tag}_train_steps.csv"
    confusion_path = out_dir / f"{tag}_best_confusion.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
    best_confusion = None
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'train_acc':>9} "
        f"{'val_loss':>9} {'val_acc':>8} {'val_top5':>8} {'macroF1':>8}"
    )
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_sampler, criterion, optimizer, device,
            epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()
        metrics, confusion = evaluate(
            model, val_loader, val_sampler, eval_criterion, device, num_classes
        )

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        flag = ""
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            best_confusion = confusion
            flag = " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": n_chans,
                    "n_samples": n_samples,
                    "num_classes": num_classes,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            save_confusion_csv(confusion_path, confusion, class_names, num_classes)

        print(
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} {train_acc:>9.4f} "
            f"{metrics['val_loss']:>9.4f} {metrics['val_acc']:>8.4f} "
            f"{metrics['val_top5']:>8.4f} {metrics['val_macro_f1']:>8.4f}{flag}"
        )

        # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "val_acc": f"{metrics['val_acc']:.6f}",
            "val_top5": f"{metrics['val_top5']:.6f}",
            "val_macro_f1": f"{metrics['val_macro_f1']:.6f}",
            "n_val": metrics["n_val"],
            "n_classes_val": metrics["n_classes_val"],
            "is_best": int(improved),
        })

    # 随机基线：均匀猜 -> top-1 = 1/C，top-5 = min(5, C)/C，macro-F1 ≈ 1/C
    random_top5 = min(5, num_classes) / num_classes
    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}\n"
        f"       val 样本 {best_metrics['n_val']} 条 / 出现 {best_metrics['n_classes_val']} 类"
        f"（全局 {num_classes} 类）\n"
        f"       随机基线 top-1={1 / num_classes:.4f} top-5={random_top5:.4f} "
        f"macroF1≈{1 / num_classes:.4f}"
    )

    if best_confusion is not None:
        print("       val 召回率最低的类别：")
        for cid, cname, recall, support in worst_classes(
            best_confusion, class_names, num_classes
        ):
            print(f"         {cid:>2} {cname:<12} recall={recall:.3f} support={support}")

    print(
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}\n"
        f"       steps:      {train_steps_path}\n"
        f"       confusion:  {confusion_path}\n"
        f"       classes:    {class_names_path}"
    )


if __name__ == "__main__":
    main()
