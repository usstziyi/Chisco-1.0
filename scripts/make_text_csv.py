"""遍历 derivatives/preprocessed_pkl，导出句表 CSV。

输出：
    datasets/preprocessed/sub-01_reading.csv   （对应 task-read，阅读阶段）
    datasets/preprocessed/sub-01_imagine.csv   （对应 task-imagine，想象阶段）

列：
    index        全表内从 0 递增的全局序号，顺序与 pkl 列表顺序一致
    day          实验天数，每 9 个 run 记作 1 天（run 1-9 -> day 1，10-18 -> day 2，…）
    run          run 编号（1 ~ 45）
    sentence     句子原文
    n_chars      非标点字符个数（按 Unicode 类别 P* 剔除标点；空白也不计入）
    type_number  句子语义类别 id（0 ~ 38），查 json/textmaps.json 得到
    type_name    语义类别名称（中文），用 type_number 查 json/classnumber.json 得到

用法：
    python scripts/make_text_csv.py
    python scripts/make_text_csv.py --pkl-dir <目录> --out-dir <目录>
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "ds005170-download"
DEFAULT_PKL_DIR = DATASET_ROOT / "derivatives" / "preprocessed_pkl"
DEFAULT_TEXTMAPS = DATASET_ROOT / "json" / "textmaps.json"
DEFAULT_CLASSNUMBER = DATASET_ROOT / "json" / "classnumber.json"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "preprocessed"

# 任务名 -> 输出文件名后缀（保持用户指定的 reading / imagine 命名）
TASK_LABEL = {"read": "reading", "imagine": "imagine"}

# 每 9 个 run 划分为 1 天（该数据集每人每天 9 个 block，共 5 天 / 45 个 run）
RUNS_PER_DAY = 9
CSV_FIELDS = ["index", "day", "run", "sentence", "n_chars", "type_number", "type_name"]


def count_n_chars(sentence: str) -> int:
    """统计非标点字符个数：排除 Unicode 标点类（P*）与空白字符。"""
    return sum(
        1
        for ch in sentence
        if not ch.isspace() and unicodedata.category(ch)[0] != "P"
    )


def parse_run(file_name: str) -> int:
    """从 sub-01_task-read_run-010_eeg.pkl 这类文件名中取出 run 编号。"""
    token = [p for p in file_name.split("_") if p.startswith("run-")][0]
    return int(token.removeprefix("run-"))


def collect(pkl_dir: Path, subject: str) -> dict[str, list[Path]]:
    """按任务收集某个被试的 pkl 文件。"""
    by_task: dict[str, list[Path]] = {}
    # 文件名是 run-01 / run-010 这类非等宽命名，字符串排序会错位，故按 run 编号数值排序
    for path in sorted(
        pkl_dir.rglob(f"{subject}_task-*_eeg.pkl"), key=lambda p: (p.name.split("task-")[1].split("_")[0], parse_run(p.name))
    ):
        task = path.name.split("task-")[1].split("_")[0]
        by_task.setdefault(task, []).append(path)
    return by_task


def write_csv(
    task: str, files: list[Path], out_path: Path, textmaps: dict[str, int], classnames: dict[int, str]
) -> list[dict]:
    """把某个任务下所有 run 的句子写成一张 CSV，返回写入的行。"""
    rows: list[dict] = []
    index = 0
    for path in files:
        run = parse_run(path.name)
        day = (run - 1) // RUNS_PER_DAY + 1
        with open(path, "rb") as f:
            items = pickle.load(f)
        for item in items:
            sentence = item["text"]
            type_number = textmaps.get(sentence, "")
            rows.append(
                {
                    "index": index,
                    "day": day,
                    "run": run,
                    "sentence": sentence,
                    "n_chars": count_n_chars(sentence),
                    "type_number": type_number,
                    "type_name": classnames.get(type_number, ""),
                }
            )
            index += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="从 preprocessed_pkl 导出句表 CSV")
    parser.add_argument("--pkl-dir", type=Path, default=DEFAULT_PKL_DIR, help="preprocessed_pkl 目录")
    parser.add_argument("--textmaps", type=Path, default=DEFAULT_TEXTMAPS, help="textmaps.json 路径（句子 -> 类别 id）")
    parser.add_argument(
        "--classnumber", type=Path, default=DEFAULT_CLASSNUMBER, help="classnumber.json 路径（类别 id -> 类别名）"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="CSV 输出目录")
    parser.add_argument("--subject", default="sub-01", help="被试标识，默认 sub-01")
    args = parser.parse_args()

    with open(args.textmaps, encoding="utf-8") as f:
        textmaps = json.load(f)
    with open(args.classnumber, encoding="utf-8") as f:
        # classnumber.json 的 key 是字符串形式的 id，转成 int 便于直接用 type_number 查
        classnames = {int(k): v for k, v in json.load(f).items()}

    by_task = collect(args.pkl_dir, args.subject)
    if not by_task:
        print(f"[错误] 在 {args.pkl_dir} 下没找到 {args.subject} 的 pkl 文件")
        return 1

    for task, files in sorted(by_task.items()):
        label = TASK_LABEL.get(task, task)
        out_path = args.out_dir / f"{args.subject}_{label}.csv"
        rows = write_csv(task, files, out_path, textmaps, classnames)

        n_chars = [r["n_chars"] for r in rows]
        runs = sorted({r["run"] for r in rows})
        days = sorted({r["day"] for r in rows})
        missing = [r["sentence"] for r in rows if r["type_number"] == ""]
        no_name = [r for r in rows if r["type_name"] == ""]
        types = sorted({r["type_number"] for r in rows if r["type_number"] != ""})
        print(f"[{task:8s}] {len(files):2d} 个 run -> {out_path}")
        print(
            f"           句子数 {len(rows)} | run 范围 {runs[0]}~{runs[-1]} | "
            f"n_chars min/mean/max = {min(n_chars)}/{sum(n_chars) / len(n_chars):.1f}/{max(n_chars)}"
        )
        print(f"           day 覆盖 {days}（每天 9 个 run）")
        print(f"           type_number: 命中 {len(rows) - len(missing)}/{len(rows)}，共 {len(types)} 类（{types[0]}~{types[-1]}）")
        if missing:
            print(f"           [警告] {len(missing)} 条句子在 textmaps.json 中查不到；示例 {missing[:3]}")
        if no_name:
            print(f"           [警告] {len(no_name)} 条 type_number 在 classnumber.json 中查不到；示例 {no_name[0]['type_number']}")
        # 提示潜在的空字符 / 多余空白，便于核对
        odd = [r for r in rows if " " in r["sentence"] or "\u3000" in r["sentence"]]
        if odd:
            print(f"           注意：{len(odd)} 条句子含空白字符，已计入剔除；示例 {odd[0]['sentence']!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
