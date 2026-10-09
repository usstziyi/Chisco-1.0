"""按 subject / task 加载 Chisco preprocessed_pkl 中的 EEG 数据。

数据来源:
    chisco/derivatives/preprocessed_pkl/
        sub-XX/eeg/
            sub-XX_task-{task}_run-*_eeg.pkl

官方每个 pkl 为 list[dict]，每个元素大致为:
    {
        "text": str,
        "input_features": ndarray,   # 通常 shape=(1, 125, 1651)
    }

    input_features 的 125 个通道并非全是 EEG：
        前 122 个是 EEG，末尾 3 个是 2×EOG + 1×STIM
        （末尾 3 个的具体顺序随被试 / run 变化，EEG 始终在前 122 个）。
    保存 NPZ 时只保留这 122 个 EEG 通道。

本脚本会在加载时额外补充:
    {
        "run": int,
        "trial": int,
    }

保存 NPZ 前会做一步清洗:
    同一天内 text 相同的样本只保留第一次出现的那条。
    load_days / load / load_all / save_days 均适用。

用法:
    uv run python scripts/prepare_dataset.py --task read
    uv run python scripts/prepare_dataset.py --subject 01 --task imagine --save-days
    uv run python scripts/prepare_dataset.py --subject 01 --task read --save-days

作为模块导入:
    from prepare_dataset import load, load_all, load_days, save_days

    records = load("01", "read")
    data = load_all("imagine")

    # 根据 run 编号推导 day:
    # day-01: run 01-09
    # day-02: run 10-18
    # day-03: run 19-27
    # ...
    #
    # 保存为:
    # datasets/{subject}/day-XX_task-{task}.npz
    #
    # 每个 npz 包含:
    #   X     : (N, C, T), float32
    #   y     : (N,), sentence text
    #   run   : (N,), int32
    #   trial : (N,), int32
    save_days("01", "read")
"""

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Paths / constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "chisco" / "derivatives" / "preprocessed_pkl"

DATASETS_ROOT = PROJECT_ROOT / "datasets"

RUNS_PER_DAY = 9

# 官方 pkl 的 input_features 通道布局（已核对 sub-01..05）:
#   共 125 个通道 = 122 EEG + 2 EOG + 1 STIM，
#   其中 EEG 始终是前 122 个，末尾 3 个的排列顺序随被试 / run 变化。
# 保存 NPZ 时只保留 EEG 通道。
N_PKL_CHANNELS = 125
N_EEG_CHANNELS = 122


# ============================================================
# Subject helpers
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


def all_subjects() -> list[str]:
    """返回 DATA_ROOT 下的全部 subject，按编号排序。"""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"数据目录不存在: {DATA_ROOT}")
    subjects = [p.name for p in DATA_ROOT.glob("sub-*") if p.is_dir()]
    def subject_number(name: str) -> int:
        match = re.fullmatch(r"sub-(\d+)", name)
        if match is None:
            return 10**9
        return int(match.group(1))

    return sorted(subjects, key=subject_number)


# ============================================================
# Run / day helpers
# ============================================================

def run_number(path: Path) -> int:
    """从文件名中提取 run 序号。

    例如:
        sub-01_task-read_run-01_eeg.pkl  -> 1
        sub-01_task-read_run-010_eeg.pkl -> 10
    """
    match = re.search(r"run-(\d+)", path.name)
    if match is None:
        raise ValueError(f"无法从文件名解析 run 编号: {path.name}")
    return int(match.group(1))


def day_number(run: int) -> int:
    """根据 run 编号计算 day。

    Chisco 1.0:
        day-01: run 01-09
        day-02: run 10-18
        day-03: run 19-27
        day-04: run 28-36
        day-05: run 37-45
    """
    if run <= 0:
        raise ValueError(f"run 编号必须 > 0，收到: {run}")
    return (run - 1) // RUNS_PER_DAY + 1


def list_run_files(subject: str, task: str) -> list[Path]:
    """返回指定 subject/task 的全部 PKL 文件，按 run 数值升序。"""
    subject = normalize_subject(subject)
    if task not in {"read", "imagine"}:
        raise ValueError(f"非法 task: {task!r}，只支持 'read' 或 'imagine'")
    eeg_dir = DATA_ROOT / subject / "eeg"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG 目录不存在: {eeg_dir}")
    files = list(eeg_dir.glob(f"{subject}_task-{task}_run-*_eeg.pkl"))
    return sorted(files, key=run_number)


def day_chunks(subject: str, task: str) -> list[tuple[int, list[Path]]]:
    """根据实际 run 编号按天分组。

    返回:
        [
            (1, [run-01, ..., run-09]),
            (2, [run-10, ..., run-18]),
            ...
        ]

    注意:
        不是机械地每 9 个文件切一组。

        如果 run-05 缺失:
            day-01 仍然只包含属于 day-01 的 run；
            run-10 不会被错误补进 day-01。
    """
    files = list_run_files(subject, task)
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        run = run_number(path)
        day = day_number(run)
        grouped[day].append(path)
    return [
        (day, sorted(paths, key=run_number))
        for day, paths in sorted(grouped.items())
    ]


# ============================================================
# PKL reading
# ============================================================

def read_run(path: Path) -> list[dict]:
    """读取单个 run，并补充 run / trial 信息。"""
    run = run_number(path)
    with path.open("rb") as f:
        raw_records = pickle.load(f)
    if not isinstance(raw_records, list):
        raise TypeError(
            f"{path} 内容应为 list，实际为 {type(raw_records).__name__}"
        )
    records: list[dict] = []
    for trial_idx, record in enumerate(raw_records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"{path} 中第 {trial_idx} 条记录不是 dict")
        if "text" not in record:
            raise KeyError(f"{path} 中第 {trial_idx} 条记录缺少 'text'")
        if "input_features" not in record:
            raise KeyError(
                f"{path} 中第 {trial_idx} 条记录缺少 'input_features'"
            )
        records.append({**record, "run": run, "trial": trial_idx})
    return records


def read_runs(paths: list[Path]) -> list[dict]:
    """按给定顺序读取并合并多个 run。"""
    records: list[dict] = []
    for path in paths:
        records.extend(read_run(path))
    return records


def dedup_text(records: list[dict]) -> list[dict]:
    """按 text 去重，只保留第一次出现的样本。

    作用范围是传进来的这批 records（即一个 day），跨天的重复不在这里处理。
    """
    seen: set[str] = set()
    kept: list[dict] = []
    for record in records:
        text = str(record["text"])
        if text in seen:
            continue
        seen.add(text)
        kept.append(record)
    return kept


# ============================================================
# Public loading API
# ============================================================

def load_days(subject: str, task: str) -> list[list[dict]]:
    """按天加载指定 subject/task。

    每个 day 内部按 text 去重，只保留第一次出现的样本。

    返回:
        [
            day_1_records,
            day_2_records,
            ...
        ]

    每条 record 包含:
        text
        input_features
        run
        trial
    """
    subject = normalize_subject(subject)
    return [
        dedup_text(read_runs(paths))
        for _, paths in day_chunks(subject, task)
    ]


def load(subject: str, task: str) -> list[dict]:
    """加载指定 subject/task 的全部 run。

    每个 day 内部按 text 去重（只保留第一次出现的样本），再按:
        day -> run -> trial

    的顺序合并为一个 list[dict]。
    """
    records: list[dict] = []
    for _, paths in day_chunks(subject, task):
        records.extend(dedup_text(read_runs(paths)))
    return records


def load_all(task: str) -> dict[str, list[dict]]:
    """加载全部 subject 的指定 task 数据。

    返回:
        {
            "sub-01": [...],
            "sub-02": [...],
            ...
        }
    """
    return {subject: load(subject, task) for subject in all_subjects()}


# ============================================================
# EEG processing for NPZ
# ============================================================

def prepare_eeg(input_features: np.ndarray) -> np.ndarray:
    """把官方 PKL 中单条 EEG 转成 (C, T)，并只保留 EEG 通道。

    官方数据通常:
        (1, 125, 1651)

    通道布局（已核对）:
        前 122 个通道是 EEG，末尾 3 个是 2×EOG + 1×STIM
        （末尾 3 个的具体顺序随被试 / run 变化）。

    本函数去掉最前面的 singleton epoch 维，丢弃末尾 3 个非 EEG 通道，
    不裁时间，不改变物理单位。

    输出:
        (122, 1651)

    dtype:
        float32
    """
    x = np.asarray(input_features)
    if x.ndim != 3:
        raise ValueError(
            f"input_features 应为 3 维，例如 (1, C, T)，实际 shape={x.shape}"
        )
    if x.shape[0] != 1:
        raise ValueError(f"input_features 第 0 维应为 1，实际 shape={x.shape}")
    x = np.squeeze(x, axis=0)
    if x.shape[0] != N_PKL_CHANNELS:
        raise ValueError(
            f"通道数应为 {N_PKL_CHANNELS}（{N_EEG_CHANNELS} EEG + 3 非 EEG），"
            f"实际 {x.shape[0]}"
        )
    x = x[:N_EEG_CHANNELS]  # 只保留前 122 个 EEG 通道
    return x.astype(np.float32, copy=False)


# ============================================================
# Save
# ============================================================

def save_days(subject: str, task: str) -> list[Path]:
    """按天保存为 NPZ。

    同一天内 text 重复的样本只保留第一次出现的那条。

    输出目录:
        datasets/{subject}/

    文件:
        day-XX_task-{task}.npz

    每个 NPZ 包含:
        X:
            shape = (N, C, T)
            dtype = float32

        y:
            shape = (N,)
            sentence text

        run:
            shape = (N,)
            dtype = int32

        trial:
            shape = (N,)
            dtype = int32
    """
    subject = normalize_subject(subject)
    out_dir = DATASETS_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for day, chunk in day_chunks(subject, task):
        records = read_runs(chunk)
        if not records:
            continue
        # 同一天内 text 重复的样本只保留第一次出现的
        records = dedup_text(records)
        # EEG
        eeg_list = [prepare_eeg(r["input_features"]) for r in records]
        # 确认所有 trial shape 一致
        shapes = {x.shape for x in eeg_list}
        if len(shapes) != 1:
            raise ValueError(
                f"{subject} / {task} / day-{day:02d} "
                f"EEG shape 不一致: {sorted(shapes)}"
            )
        X = np.stack(eeg_list, axis=0)
        # Text
        y = np.array([str(r["text"]) for r in records])
        # Provenance
        runs = np.array([r["run"] for r in records], dtype=np.int32)
        trials = np.array([r["trial"] for r in records], dtype=np.int32)
        # Save
        path = out_dir / f"day-{day:02d}_task-{task}.npz"
        np.savez(path, X=X, y=y, run=runs, trial=trials)
        saved_paths.append(path)
        del records
        del eeg_list
        del X
        del y
        del runs
        del trials
    return saved_paths


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 subject/task 加载 Chisco preprocessed_pkl EEG 数据"
    )
    parser.add_argument(
        "--subject",
        help="被试编号，例如 01、1 或 sub-01；不传则处理全部被试",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["imagine", "read"],
        help="任务类型",
    )
    parser.add_argument(
        "--save-days",
        action="store_true",
        help="根据 run 编号按天保存到 datasets/{subject}/day-XX_task-{task}.npz",
    )
    args = parser.parse_args()
    subjects = (
        [normalize_subject(args.subject)] if args.subject else all_subjects()
    )
    for subject in subjects:
        print(f"\n{'=' * 60}\n{subject} / {args.task}\n{'=' * 60}")
        chunks = day_chunks(subject, args.task)
        if not chunks:
            print("没有找到对应的 PKL 文件")
            continue
        # 显示每天包含哪些 run
        for day, chunk in chunks:
            runs = [run_number(p) for p in chunk]
            expected_start = (day - 1) * RUNS_PER_DAY + 1
            expected_end = day * RUNS_PER_DAY
            expected = set(range(expected_start, expected_end + 1))
            actual = set(runs)
            missing = sorted(expected - actual)
            print(f"day-{day:02d}: {','.join(map(str, runs))}")
            if missing:
                print("  缺失 run: " + ",".join(map(str, missing)))
        # 保存 / 仅统计
        if args.save_days:
            paths = save_days(subject, args.task)
            for path in paths:
                with np.load(path) as data:
                    print(f"已保存: {path}")
                    print(f"  X: {data['X'].shape}, {data['X'].dtype}")
                    print(f"  y: {data['y'].shape}")
                    print(f"  run: {data['run'].min()}-{data['run'].max()}")
            print(f"{subject} / {args.task}: 已保存 {len(paths)} 天")
        else:
            total_records = 0
            for day, chunk in chunks:
                records = read_runs(chunk)
                print(f"day-{day:02d}: {len(records)} 条样本")
                total_records += len(records)
                del records
            print(
                f"{subject} / {args.task}: "
                f"{len(chunks)} 天，共 {total_records} 条样本"
            )


if __name__ == "__main__":
    main()
