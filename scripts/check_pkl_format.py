"""校验 preprocessed_pkl 中 EEG pkl 文件格式。

每个文件为 list[dict]，列表元素为 {"text": str, "input_features": ndarray}。

用法:
    uv run python scripts/check_pkl_format.py --subject 01 --task read
    uv run python scripts/check_pkl_format.py --subject sub-01 --task imagine
    uv run python scripts/check_pkl_format.py --task read          # 不传 subject 则遍历所有被试
"""

import argparse
import csv
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

EXPECTED_KEYS = {"text", "input_features"}
CSV_FIELDS = ["file", "task", "status", "records", "features_shape", "features_dtype", "errors_count", "errors"]


def load_and_check(path: Path) -> tuple[list | None, list[str]]:
    with path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        return None, [f"顶层应为 list，实际为 {type(data).__name__}"]

    errors: list[str] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            errors.append(f"[{i}] 应为 dict，实际为 {type(rec).__name__}")
            continue
        if set(rec) != EXPECTED_KEYS:
            errors.append(f"[{i}] 键应为 {sorted(EXPECTED_KEYS)}，实际为 {sorted(rec)}")
        if not isinstance(rec.get("text"), str):
            errors.append(f"[{i}] text 应为 str，实际为 {type(rec.get('text')).__name__}")
        if not hasattr(rec.get("input_features"), "shape"):
            errors.append(f"[{i}] input_features 应为数组，实际为 {type(rec.get('input_features')).__name__}")
    return data, errors


def scan(eeg_dir: Path, subject: str, task: str) -> tuple[list[dict], int, int, Counter]:
    """校验单个被试的 eeg 目录，返回 (csv 行, 通过数, 总数, 形状计数)。"""
    # 文件名 run 号是「数字前补0」（run-01 ... run-045），字符串排序会乱，按数字排序
    files = sorted(
        eeg_dir.glob(f"{subject}_task-{task}_run-*_eeg.pkl"),
        key=lambda p: int(re.search(r"run-(\d+)", p.name).group(1)),
    )

    rows: list[dict] = []
    ok, shapes = 0, Counter()
    for path in files:
        data, errors = load_and_check(path)
        if errors:
            print(f"[FAIL] {path.name}: {len(errors)} 处问题")
            for err in errors[:5]:
                print(f"       - {err}")
            if len(errors) > 5:
                print(f"       - ... 其余 {len(errors) - 5} 处省略")
        else:
            ok += 1
            print(f"[ OK ] {path.name}: {len(data)} 条记录")

        file_shapes: Counter = Counter()
        for rec in data or []:
            feats = rec.get("input_features") if isinstance(rec, dict) else None
            if hasattr(feats, "shape"):
                file_shapes[(tuple(feats.shape), feats.dtype.str)] += 1
        shapes.update(file_shapes)

        rows.append({
            "file": path.name,
            "task": task,
            "status": "FAIL" if errors else "OK",
            "records": len(data) if isinstance(data, list) else 0,
            "features_shape": "; ".join(str(shape) for shape, _ in file_shapes),
            "features_dtype": "; ".join(dtype for _, dtype in file_shapes),
            "errors_count": len(errors),
            "errors": " | ".join(errors),
        })

    return rows, ok, len(files), shapes


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 EEG pkl 文件格式")
    parser.add_argument("--subject", help="被试编号，如 01 或 sub-01；不传则遍历所有被试")
    parser.add_argument("--task", required=True, choices=["imagine", "read"], help="任务类型")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "chisco/derivatives/preprocessed_pkl"

    if args.subject:
        subjects = [args.subject if args.subject.startswith("sub-") else f"sub-{int(args.subject):02d}"]
    else:
        subjects = sorted(p.name for p in data_root.glob("sub-*") if p.is_dir())
    if not subjects:
        print(f"未找到被试目录: {data_root / 'sub-*'}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    total_files = total_ok = 0
    all_shapes: Counter = Counter()
    for subject in subjects:
        eeg_dir = data_root / subject / "eeg"
        if not eeg_dir.is_dir():
            print(f"[SKIP] {subject}: 缺少 eeg 目录 {eeg_dir}", file=sys.stderr)
            continue
        print(f"=== {subject} ({args.task}) ===")
        sub_rows, ok, total, shapes = scan(eeg_dir, subject, args.task)
        if not total:
            print(f"[SKIP] {subject}: 未找到 pkl 文件", file=sys.stderr)
            continue
        rows.extend(sub_rows)
        total_files += total
        total_ok += ok
        all_shapes.update(shapes)

    if not total_files:
        print("未找到任何 pkl 文件", file=sys.stderr)
        return 2

    print(f"\n共 {total_files} 个文件，通过 {total_ok}，失败 {total_files - total_ok}")
    if all_shapes:
        print("input_features 形状分布 (shape, dtype) -> 次数:")
        for (shape, dtype), count in all_shapes.most_common():
            print(f"  {shape} {dtype} -> {count}")

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = subjects[0] if args.subject else "all"
    out_path = out_dir / f"check_{tag}_{args.task}.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果已写入: {out_path}")

    return 0 if total_ok == total_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
