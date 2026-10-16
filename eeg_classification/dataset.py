"""Chisco（OpenNeuro ds005170）EEG 语义类别分类 Dataset。

与 ``baseline.dataset.ChiscoEEGDataset`` 用同一批数据（同一张 CSV、同一批 pkl），
区别只在**标签**与**样本组织**：

- feature：该行的整句 EEG ``(C, T)``，``C = 122``（仅 EEG 通道）；
- label：该行句子的 ``type_number``（语义类别 id，0~38，共 39 类），
  查 ``json/textmaps.json`` 得到；这里直接从 CSV 的 ``type_number`` 列读。

    CSV 一行 = 一句话（一个试次）
        ↓  用 (run, pos) 取回 EEG 片段
    (eeg, label)
      eeg    torch.float32 (122, T)，微伏
      label  torch.long    标量，0 ~ N_CLASSES-1

``T`` 由 task 决定、与句长无关（``n_chars`` 只用于过滤/分析，不参与切分）：
reading 2501（0~5.0 s）、imagine 1651（5.0~8.3 s），采样率 500 Hz。

定位方式（CSV 的 ``run`` / ``pos`` 两列）：

- ``run`` -> ``preprocessed_pkl/<subject>/eeg/<subject>_task-<pkl_task>_run-<NNN>_eeg.pkl``
- ``pos`` -> 该 pkl 内 ``list`` 的下标，每项形如
  ``{"text": str, "input_features": (1, 125, T)}``

通道：pkl 的 125 通道 = 122 EEG + ``VEO`` + ``HEO`` + ``Trigger``，本 Dataset 只取前 122 个。
``Trigger`` 承载 65280 量级的事件 marker，和 VEO/HEO 一并剔除。

单位：pkl 里的采样值沿用 MNE 的默认约定，即**伏特（V）**；这里统一乘
``VOLT_TO_MICROVOLT`` 换算成**微伏（µV）**再返回，理由见 ``baseline/dataset.py``。

类别数
------
``type_number`` 是**全局**标签空间（``json/textmaps.json`` 覆盖全部被试），
所以对每个 ``(subject, task)`` 都用同一个 ``num_classes``，跨被试可直接比指标。
``num_classes`` 由 CSV 里出现过的最大 ``type_number + 1`` 推出（实践中为 39）。

``textmaps.json`` 里查不到类别的行（``type_number`` 为空）会被直接丢弃，
否则它们既不能训练也不能评测。

``__getitem__`` 返回 ``(eeg, label)``；句子原文仍保留在 ``rows`` 里，
供训练脚本按句子切分 train / val（同一句话的多次试次不能横跨两侧）。
"""

from __future__ import annotations

import csv
import json
import pickle
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL_ROOT = PROJECT_ROOT / "datasets/ds005170-download/derivatives/preprocessed_pkl"
CSV_ROOT = PROJECT_ROOT / "datasets/preprocessed"
CLASSNUMBER_PATH = PROJECT_ROOT / "datasets/ds005170-download/json/classnumber.json"

SFREQ = 500
# pkl 前 122 个通道是 EEG，后 3 个依次为 VEO / HEO / Trigger
N_EEG_CHANNELS = 122
# pkl 里的采样值是伏特（MNE 约定），统一换算成微伏再返回
VOLT_TO_MICROVOLT = 1e6

# 传入的 task -> (CSV 文件名后缀, pkl 文件名里的 task 字段)
TASK_ALIASES = {
    "reading": ("reading", "read"),
    "imagine": ("imagine", "imagine"),
}

_FILE_PATTERN = re.compile(r"_task-(?P<task>[a-z]+)_run-(?P<run>\d+)_eeg\.pkl$")


def load_class_names(path: Path = CLASSNUMBER_PATH) -> dict[int, str]:
    """读 ``json/classnumber.json``，返回 ``{type_number: 类别名}``（仅用于报告）。"""
    with Path(path).open(encoding="utf-8") as handle:
        return {int(k): v for k, v in json.load(handle).items()}


class ChiscoEEGClassificationDataset(Dataset):
    """一个 ``(subject, task)`` 组合对应一张 CSV，构成一个分类 Dataset。

    Args:
        subject: 被试编号，如 ``sub-01``。
        task: ``reading`` / ``imagine``。
        csv_root: ``make_text_csv.py`` 输出的 CSV 目录。
        pkl_root: ``derivatives/preprocessed_pkl`` 目录。
        cache_size: 缓存的 run 个数。单个 run 的 pkl 解包后约 444 MB，
            故默认只留 2 个；CSV 行序按 run 递增，顺序遍历时命中率接近 100%，
            随机打乱访问则会反复重载。

    注意：CSV 中同一句话可能在不同 run 里重复出现（同一句子跑多个试次），
    这里不去重，每条记录都对应各自试次的 EEG；重复的试次也因此共享同一 label。
    """

    def __init__(
        self,
        subject: str,
        task: str,
        csv_root: Path = CSV_ROOT,
        pkl_root: Path = PKL_ROOT,
        cache_size: int = 2,
    ):
        if task.lower() not in TASK_ALIASES:
            raise ValueError(f"task 需为 reading/imagine 之一，收到 {task!r}")
        label, pkl_task = TASK_ALIASES[task.lower()]

        self.subject = subject
        self.task = label
        self.pkl_task = pkl_task

        csv_path = Path(csv_root) / f"{subject}_{label}.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            raw_rows: list[dict] = list(csv.DictReader(handle))

        # 丢掉 textmaps.json 里查不到类别的行（type_number 为空 → 无标签可用）
        self.rows: list[dict] = [r for r in raw_rows if str(r.get("type_number", "")).strip() != ""]
        self.n_dropped_unlabeled = len(raw_rows) - len(self.rows)
        if not self.rows:
            raise ValueError(f"{csv_path} 里没有带 type_number 的行")

        # 每行的标签，提前转成张量，__getitem__ 里只做索引
        self.labels = torch.tensor(
            [int(r["type_number"]) for r in self.rows], dtype=torch.long
        )

        # run -> pkl 路径。文件名是 run-01 / run-010 这种非等宽命名，
        # 直接按名字排序会错位，这里按解析出的 run 编号建索引。
        self.eeg_dir = Path(pkl_root) / subject / "eeg"
        self._pkl_paths: dict[int, Path] = {}
        for path in self.eeg_dir.glob(f"{subject}_task-{pkl_task}_run-*_eeg.pkl"):
            match = _FILE_PATTERN.search(path.name)
            if match is not None:
                self._pkl_paths[int(match["run"])] = path

        self._cache: OrderedDict[int, list[dict]] = OrderedDict()
        self._cache_size = cache_size

    @property
    def num_classes(self) -> int:
        """全局类别数 = 本 CSV 中出现过的最大 type_number + 1。"""
        return int(self.labels.max().item()) + 1

    def class_counts(self, indices=None) -> np.ndarray:
        """按类别统计样本数，返回长度 ``num_classes`` 的数组。"""
        labels = (
            self.labels.numpy()
            if indices is None
            else self.labels[np.asarray(indices, dtype=np.int64)].numpy()
        )
        return np.bincount(labels, minlength=self.num_classes)

    def __len__(self) -> int:
        return len(self.rows)

    def _items(self, run: int) -> list[dict]:
        """取某个 run 的 pkl 内容（带 LRU 缓存）。"""
        if run in self._cache:
            self._cache.move_to_end(run)
            return self._cache[run]

        path = self._pkl_paths.get(run)
        if path is None:
            raise FileNotFoundError(
                f"{self.subject} {self.task} 找不到 run-{run} 的 pkl（目录 {self.eeg_dir}）"
            )
        with path.open("rb") as handle:
            items: list[dict] = pickle.load(handle)

        self._cache[run] = items
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return items

    def __getitem__(self, index: int):
        row = self.rows[index]
        item = self._items(int(row["run"]))[int(row["pos"])]

        # input_features: (1, 125, T) -> 去掉 batch 维、只留 EEG 通道 -> (122, T)
        eeg = np.asarray(item["input_features"][0][:N_EEG_CHANNELS], dtype=np.float32)
        # 伏特 -> 微伏。这里刻意用 `*=` 之外的新数组：若 pkl 本身是 float32，
        # np.asarray 会返回视图，原地乘会把 _items() 缓存里的数据也改掉。
        eeg = eeg * VOLT_TO_MICROVOLT
        return torch.from_numpy(eeg), self.labels[index]


def main() -> None:
    """冒烟检查：遍历所有 CSV 建立 dataset，打印规模与类别分布，并跑一个 batch。"""
    names = load_class_names()
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, task = csv_path.stem.split("_", 1)
        dataset = ChiscoEEGClassificationDataset(subject, task)
        eeg, label = dataset[0]
        counts = dataset.class_counts()
        present = np.flatnonzero(counts)
        print(
            f"{subject} {task:<8} {len(dataset):>5} 条（丢弃无标签 {dataset.n_dropped_unlabeled}）"
            f" | 首条 {tuple(eeg.shape)} {eeg.dtype} 取值 [{eeg.min():.3e}, {eeg.max():.3e}]"
            f" | label={int(label)} ({names.get(int(label), '?')})"
        )
        print(
            f"         类别数 num_classes={dataset.num_classes}"
            f" | 出现 {len(present)} 类"
            f" | 每类 min/mean/max = {counts[present].min()}"
            f"/{counts[present].mean():.1f}/{counts[present].max()}"
        )

    first = ChiscoEEGClassificationDataset("sub-01", "reading")
    loader = DataLoader(first, batch_size=4)
    eeg, labels = next(iter(loader))
    print(f"\nsub-01 reading 取一个 batch (batch_size=4):")
    print(f"  eeg    {tuple(eeg.shape)} {eeg.dtype}")
    print(f"  labels {labels.tolist()} {labels.dtype}")


if __name__ == "__main__":
    main()
