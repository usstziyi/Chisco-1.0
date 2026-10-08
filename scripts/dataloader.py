"""构建 Chisco-1.0 的 PyTorch Dataset / DataLoader。

数据来源:
    datasets/{subject}/day-XX_task-{task}.npz

    每个 npz 由 prepare_dataset.py --save-days 生成，包含:
        X     : (N, C, T), float32
        y     : (N,), 句子文本
        run   : (N,), int32
        trial : (N,), int32

划分策略:
    在「单个 day 内部」随机划分 train / val。
    同一个 day 的 trial 按固定 seed 打乱后，取 val_ratio 比例作为验证集。

样本格式 (每条 __getitem__ 返回 dict):
    return_meta=False (默认，纯训练，只含 eeg/text):
        {
            "eeg":   FloatTensor, shape=(C, T)  # 默认 (125, 1651)
            "text":  str                        # 原始句子，留给训练脚本自行用 LaBSE 编码
        }

    return_meta=True (附带 run/trial 元信息):
        {
            "eeg":   FloatTensor, shape=(C, T)
            "text":  str
            "run":   int
            "trial": int
        }

    DataLoader 默认 collate 后 (return_meta=False):
        "eeg":   (B, C, T)
        "text":  list[str]  (长度 B)

用法:
    uv run python scripts/dataloader.py --subject 01 --task imagine --day 1

作为模块导入:
    from dataloader import ChiscoEEGDataset, build_dataloaders

    train_loader, val_loader = build_dataloaders(
        subject="01",
        task="imagine",
        day=1,
        val_ratio=0.2,
        batch_size=16,
        # return_meta=True,   # 需要 run/trial（评测/调试）时再打开
    )
    for batch in train_loader:
        eeg = batch["eeg"]        # (B, 125, 1651)
        texts = batch["text"]     # list[str]
        ...
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# Paths / constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS_ROOT = PROJECT_ROOT / "datasets"

TASKS = ("read", "imagine")


# ============================================================
# Subject / path helpers
# ============================================================

def normalize_subject(subject: str) -> str:
    """统一 subject 格式。

    支持:
        "1"      -> "sub-01"
        "01"     -> "sub-01"
        "sub-01" -> "sub-01"
    """
    subject = subject.strip()
    if subject.startswith("sub-"):
        number = subject.removeprefix("sub-")
    else:
        number = subject
    try:
        return f"sub-{int(number):02d}"
    except ValueError as e:
        raise ValueError(
            f"非法 subject: {subject!r}，应类似 '01'、'1' 或 'sub-01'"
        ) from e


def day_path(subject: str, task: str, day: int) -> Path:
    """返回指定 subject/task/day 的 npz 路径，并校验存在。"""
    subject = normalize_subject(subject)
    if task not in TASKS:
        raise ValueError(f"非法 task: {task!r}，只支持 {TASKS}")
    if day <= 0:
        raise ValueError(f"day 必须 > 0，收到: {day}")
    path = DATASETS_ROOT / subject / f"day-{day:02d}_task-{task}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {path}\n"
            "请先运行: "
            f"uv run python scripts/prepare_dataset.py --subject {subject} "
            f"--task {task} --save-days"
        )
    return path


# ============================================================
# Dataset
# ============================================================

class ChiscoEEGDataset(Dataset):
    """单个 subject / task / day 的 EEG 数据集。

    Parameters
    ----------
    subject:
        被试编号，例如 "01"、"1" 或 "sub-01"。

    task:
        "read" 或 "imagine"。

    day:
        day 编号，从 1 开始。

    return_meta:
        是否在样本中附带 run / trial 元信息。
        False (默认): 只返回 eeg / text
        True:         返回 eeg / text / run / trial

    每条样本返回 dict:
        return_meta=False:
            {
                "eeg":   FloatTensor (C, T),
                "text":  str,
            }
        return_meta=True:
            {
                "eeg":   FloatTensor (C, T),
                "text":  str,
                "run":   int,
                "trial": int,
            }
    """

    def __init__(
        self,
        subject: str,
        task: str,
        day: int,
        return_meta: bool = False,
    ):
        self.subject = normalize_subject(subject)
        self.task = task
        self.day = day
        self.return_meta = return_meta
        self.path = day_path(self.subject, task, day)

        with np.load(self.path) as data:
            self.X = data["X"]
            self.y = data["y"]
            self.run = data["run"]
            self.trial = data["trial"]

        if self.X.ndim != 3:
            raise ValueError(
                f"{self.path} 中 X 应为 3 维 (N, C, T)，实际 shape={self.X.shape}"
            )
        if not (len(self.X) == len(self.y) == len(self.run) == len(self.trial)):
            raise ValueError(
                f"{self.path} 中 X/y/run/trial 长度不一致: "
                f"{len(self.X)}/{len(self.y)}/{len(self.run)}/{len(self.trial)}"
            )

    @property
    def n_chans(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> dict:
        eeg = self.X[idx]
        sample = {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)).float(),
            "text": str(self.y[idx]),
        }
        if self.return_meta:
            sample["run"] = int(self.run[idx])
            sample["trial"] = int(self.trial[idx])
        return sample


# ============================================================
# Split
# ============================================================

def split_train_val(
    n_samples: int,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """在单个 day 内部把样本索引划成 train / val。

    返回按索引升序排列的 (train_idx, val_idx)。

    Parameters
    ----------
    n_samples:
        该 day 的样本总数。

    val_ratio:
        验证集比例，需满足 0 < val_ratio < 1。

    seed:
        随机种子，保证划分可复现。
    """
    if n_samples < 2:
        raise ValueError(f"样本数不足以划分 train/val，收到 n_samples={n_samples}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio 需满足 0 < val_ratio < 1，收到: {val_ratio}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_val = int(round(n_samples * val_ratio))
    # 保证 train / val 至少各 1 条
    n_val = min(max(n_val, 1), n_samples - 1)

    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])
    return train_idx, val_idx


# ============================================================
# DataLoader
# ============================================================

def build_dataloaders(
    subject: str,
    task: str,
    day: int,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    return_meta: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """构建单个 day 的 train / val DataLoader。

    划分在 day 内部完成，返回:
        (train_loader, val_loader)

    return_meta=False (默认) 时 batch 只含:
        "eeg":   (B, C, T)
        "text":  list[str]

    return_meta=True 时 batch 为 dict:
        "eeg":   (B, C, T)
        "text":  list[str]
        "run":   (B,)
        "trial": (B,)
    """
    dataset = ChiscoEEGDataset(subject, task, day, return_meta=return_meta)
    train_idx, val_idx = split_train_val(len(dataset), val_ratio=val_ratio, seed=seed)

    train_set = Subset(dataset, train_idx.tolist())
    val_set = Subset(dataset, val_idx.tolist())

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


# ============================================================
# CLI (冒烟测试)
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 Chisco-1.0 单 day 的 train/val DataLoader"
    )
    parser.add_argument("--subject", required=True, help="被试编号，如 01 或 sub-01")
    parser.add_argument("--task", required=True, choices=list(TASKS), help="任务类型")
    parser.add_argument("--day", type=int, required=True, help="day 编号，从 1 开始")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--meta",action="store_true",help="batch 中额外带上 run/trial（默认不带，只返回 eeg 和 text）")
    args = parser.parse_args()

    return_meta = args.meta

    dataset = ChiscoEEGDataset(
        args.subject, args.task, args.day, return_meta=return_meta
    )
    print(
        f"{dataset.subject} / {dataset.task} / day-{dataset.day:02d}\n"
        f"  文件: {dataset.path}\n"
        f"  样本数: {len(dataset)}\n"
        f"  EEG shape: ({dataset.n_chans}, {dataset.n_times})"
    )

    train_loader, val_loader = build_dataloaders(
        subject=args.subject,
        task=args.task,
        day=args.day,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        return_meta=return_meta,
    )
    print(
        f"\ntrain: {len(train_loader.dataset)} 条, {len(train_loader)} 个 batch\n"
        f"val:   {len(val_loader.dataset)} 条, {len(val_loader)} 个 batch"
    )

    batch = next(iter(train_loader))
    print(
        "\n一个 train batch:\n"
        f"  keys:  {sorted(batch.keys())}\n"
        f"  eeg:   {tuple(batch['eeg'].shape)}, {batch['eeg'].dtype}\n"
        f"  text:  {type(batch['text']).__name__} (长度 {len(batch['text'])})\n"
        f"  样例文本: {batch['text'][0]!r}"
    )
    if return_meta:
        print(
            f"  run:   {tuple(batch['run'].shape)}\n"
            f"  trial: {tuple(batch['trial'].shape)}"
        )


if __name__ == "__main__":
    main()
