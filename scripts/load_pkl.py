"""按 subject / task 加载 preprocessed_pkl 中的 EEG 数据。

数据来源: chisco/derivatives/preprocessed_pkl/{subject}/eeg/{subject}_task-{task}_run-*_eeg.pkl
每个 pkl 为 list[dict]，元素为 {"text": str, "input_features": ndarray}。

用法:
    uv run python scripts/load_pkl.py --task read
    uv run python scripts/load_pkl.py --subject 01 --task imagine

作为模块导入:
    from load_pkl import load, load_all

    records = load("01", "read")          # list[dict]，所有 run 合并
    data = load_all("imagine")            # {"sub-01": [...], "sub-02": [...]}
"""

import argparse
import pickle
import re
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "chisco/derivatives/preprocessed_pkl"


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    return subject if subject.startswith("sub-") else f"sub-{int(subject):02d}"


def list_run_files(subject: str, task: str) -> list[Path]:
    """返回该 subject/task 下的 pkl 文件，按 run 号升序（run 号是「数字前补0」，不能按字符串排）。"""
    eeg_dir = DATA_ROOT / subject / "eeg"
    files = eeg_dir.glob(f"{subject}_task-{task}_run-*_eeg.pkl")
    return sorted(files, key=lambda p: int(re.search(r"run-(\d+)", p.name).group(1)))


def load(subject: str, task: str) -> list[dict]:
    """加载单个被试指定 task 的全部 run，按 run 顺序合并为一个 list[dict]。"""
    subject = normalize_subject(subject)
    records: list[dict] = []
    for path in list_run_files(subject, task):
        with path.open("rb") as f:
            records.extend(pickle.load(f))
    return records


def load_all(task: str) -> dict[str, list[dict]]:
    """加载所有被试指定 task 的数据，返回 {subject: list[dict]}。"""
    subjects = sorted(p.name for p in DATA_ROOT.glob("sub-*") if p.is_dir())
    return {subject: load(subject, task) for subject in subjects}


def main() -> None:
    parser = argparse.ArgumentParser(description="按 subject/task 加载 EEG 数据")
    parser.add_argument("--subject", help="被试编号，如 01 或 sub-01；不传则加载所有被试")
    parser.add_argument("--task", required=True, choices=["imagine", "read"], help="任务类型")
    args = parser.parse_args()

    if args.subject:
        records = load(args.subject, args.task)
        print(f"{normalize_subject(args.subject)} / {args.task}: {len(records)} 条样本")
        if records:
            print(f"  首条 text: {records[0]['text']}")
    else:
        dataset = load_all(args.task)
        for subject, records in dataset.items():
            print(f"{subject} / {args.task}: {len(records)} 条样本")
        print(f"共 {len(dataset)} 个被试，{sum(len(r) for r in dataset.values())} 条样本")


if __name__ == "__main__":
    main()
