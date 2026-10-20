"""Chisco（OpenNeuro ds005170）中文句子级 EEG Dataset。

每个样本对应 ``datasets/preprocessed/<subject>_<task>.csv`` 里的一行（一句完整中文句子）：
以该行的 ``(run, pos)`` 从 ``derivatives/preprocessed_pkl`` 取回 EEG 片段。

``__getitem__`` 返回 ``(eeg, sentence)``：

- ``eeg``：``torch.float32``，``(C, T)``，``C = 122``（仅 EEG 通道）
- ``sentence``：该行原文句子

``T`` 由任务决定、与句长无关（``n_chars`` 只用于过滤/分析，不参与切分）：
reading 2501（0~5.0 s）、imagine 1651（5.0~8.3 s），采样率 500 Hz。

定位方式（CSV 的 ``run`` / ``pos`` 两列）：

- ``run`` -> ``preprocessed_pkl/<subject>/eeg/<subject>_task-<pkl_task>_run-<NNN>_eeg.pkl``
- ``pos`` -> 该 pkl 内 ``list`` 的下标，每项形如
  ``{"text": str, "input_features": (1, 125, T)}``

通道：pkl 的 125 通道 = 122 EEG + ``VEO`` + ``HEO`` + ``Trigger``，本 Dataset 只取前 122 个
（顺序与 fif 的 ``ch_names`` 一致）。``Trigger`` 承载 65280 量级的事件 marker，
和 VEO/HEO 一并剔除，既避免标签泄漏也避免量纲差异。

单位：pkl 里的采样值沿用 MNE 的默认约定，即**伏特（V）**；这里统一乘
``VOLT_TO_MICROVOLT`` 换算成**微伏（µV）**再返回。伏特量级约 1e-5，直接喂给网络会让
首层卷积的梯度被 1e-6 级的输入幅度缩掉（GroupNorm 只保证前向尺度不变，救不了梯度），
所以在数据出口就换算掉，模型侧不必再做缩放。
"""

from __future__ import annotations

import csv
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


class ChiscoEEGDataset(Dataset):
    """一个 ``(subject, task)`` 组合对应一张 CSV，构成一个 Dataset。

    Args:
        subject: 被试编号，如 ``sub-01``。
        task: ``reading`` / ``imagine``。
        csv_root: ``make_text_csv.py`` 输出的 CSV 目录。
        pkl_root: ``derivatives/preprocessed_pkl`` 目录。
        cache_size: 缓存的 run 个数。单个 run 的 pkl 解包后约 444 MB，
            故默认只留 2 个；CSV 行序按 run 递增，顺序遍历时命中率接近 100%，
            随机打乱访问则会反复重载。

    注意：CSV 已由 ``scripts/make_text_csv.py`` 按句子去重（每句只保留首次出现的
    试次），所以一个数据集内的 sentence 唯一；本类不做额外去重。
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
            self.rows: list[dict] = list(csv.DictReader(handle))

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
        # np.asarray 会返回视图，原地乘会把 _items() 缓存里的数据也改掉，
        # 导致同一个 run 被反复换算。
        eeg = eeg * VOLT_TO_MICROVOLT
        return torch.from_numpy(eeg), str(row["sentence"])


def main() -> None:
    """冒烟检查：遍历所有 CSV 建立 dataset，打印规模，并跑一个 batch。"""
    datasets: list[ChiscoEEGDataset] = []
    for csv_path in sorted(CSV_ROOT.glob("*.csv")):
        subject, label = csv_path.stem.split("_", 1)
        dataset = ChiscoEEGDataset(subject, label)
        datasets.append(dataset)

        eeg, sentence = dataset[0]
        print(
            f"{subject} {label:<8} {len(dataset):>5} 条 | 首条 {tuple(eeg.shape)} {eeg.dtype} | "
            f"取值 [{eeg.min():.3e}, {eeg.max():.3e}] | {sentence!r}"
        )

    first = datasets[0]
    loader = DataLoader(first, batch_size=4)
    eeg, sentences = next(iter(loader))
    print(f"\n{first.subject} {first.task} 取一个 batch (batch_size=4):")
    print(f"  eeg  {tuple(eeg.shape)} {eeg.dtype}")
    print(f"  文本 {sentences}")


if __name__ == "__main__":
    main()
