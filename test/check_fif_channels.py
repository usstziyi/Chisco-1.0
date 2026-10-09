"""遍历 test 目录下的所有 FIF，检查通道信息（名称 / 类型 / 数量）与通道顺序。

用法:
    uv run python check_fif_channels.py                 # 递归扫描 test/preprocessed_fif
    uv run python check_fif_channels.py --full          # 打印每个文件的全部通道
    uv run python check_fif_channels.py --summary       # 精简模式：只按目录汇总
    uv run python check_fif_channels.py --dir D:/path   # 指定其它目录

说明:
- 目录结构为 preprocessed_fif/{subject}/eeg/*.fif，遍历时递归查找所有 *.fif。
- 复用 plot_fif.read_any_fif，Raw / Epochs 都能读。
- 默认只显示每个文件的头尾各 3 个通道 + 所有非 EEG 通道；
  --full 时打印全部通道。
- --summary 时不再逐个文件打印，只按“所在目录”分组汇总通道布局与一致性。
- 最后按“通道序列（名称 + 类型）”对所有文件分组，检查顺序是否一致。
"""

import argparse
from collections import Counter
from pathlib import Path

from plot_fif import read_any_fif

# 预处理后的 FIF 根目录: test/preprocessed_fif/{subject}/eeg/*.fif
PREPROCESSED_ROOT = Path(__file__).resolve().parent / "preprocessed_fif"


def iter_fif_files(root):
    """递归收集 root 下所有 *.fif，按相对路径排序（顺序稳定、可复现）。"""
    return sorted(root.rglob("*.fif"))


def channel_table(fif_file):
    """读一个 FIF，返回 (kind, [(通道名, 类型), ...])。"""
    inst, kind = read_any_fif(fif_file)
    return kind, list(zip(inst.ch_names, inst.get_channel_types()))


def print_file(fif_file, kind, table, full):
    """打印单个文件的通道信息。"""
    counts = Counter(ch_type for _, ch_type in table)
    summary = ", ".join(f"{t}×{n}" for t, n in sorted(counts.items()))
    names = [name for name, _ in table]
    dup = [name for name, c in Counter(names).items() if c > 1]

    print(f"  kind   : {kind}")
    print(f"  n_chans: {len(table)}  ({summary})")
    print(f"  重复名 : {dup if dup else '无'}")

    width = max((len(name) for name, _ in table), default=4)
    for i, (name, ch_type) in enumerate(table):
        is_edge = i < 3 or i >= len(table) - 3
        if full or is_edge or ch_type != "eeg":
            print(f"    #{i:03d}  {name:<{width}}  {ch_type}")
    if not full:
        print("    ...（--full 可打印全部通道）")


def first_diff(a, b):
    """返回两个通道序列的首个差异描述。"""
    if len(a) != len(b):
        return f"通道数不同 ({len(a)} vs {len(b)})"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"首个差异在 #{i}: {x} vs {y}"
    return "完全一致"


def layout_desc(table):
    """把一条通道序列（(名称, 类型) 列表）压成一句话描述。"""
    counts = Counter(ch_type for _, ch_type in table)
    summary = ", ".join(f"{t}×{n}" for t, n in sorted(counts.items()))
    non_eeg = [f"#{i} {name}({t})" for i, (name, t) in enumerate(table) if t != "eeg"]
    tail = f"  非EEG: {', '.join(non_eeg)}" if non_eeg else ""
    return f"n_chans={len(table)} ({summary}){tail}"


def print_summary(tables):
    """精简模式：按“所在目录”分组汇总，不逐个文件打印。"""
    groups = {}
    for name, table in tables.items():
        groups.setdefault(str(Path(name).parent), []).append(table)

    print("按目录汇总")
    print("-" * 78)
    for directory in sorted(groups):
        layouts = {}
        for table in groups[directory]:
            key = tuple(table)
            layouts[key] = layouts.get(key, 0) + 1
        if len(layouts) == 1:
            desc = layout_desc(next(iter(layouts)))
        else:
            desc = f"{len(layouts)} 种布局（顺序不一致，请用默认模式排查）"
        print(f"{directory:<24} 文件 {len(groups[directory]):>3}  布局 {len(layouts)}  {desc}")

    # 总体
    overall = {}
    for table in tables.values():
        key = tuple(table)
        overall[key] = overall.get(key, 0) + 1
    print("-" * 78)
    if len(overall) == 1:
        print(f"总体：{len(tables)} 个文件通道完全一致；{layout_desc(next(iter(overall)))}")
    else:
        print(f"总体：{len(tables)} 个文件，共 {len(overall)} 种布局：")
        for key, count in overall.items():
            print(f"  ×{count}  {layout_desc(key)}")


def main():
    parser = argparse.ArgumentParser(
        description="检查目录下所有 FIF 的通道信息与通道顺序"
    )
    parser.add_argument(
        "--dir",
        default=str(PREPROCESSED_ROOT),
        help="FIF 根目录，默认 test/preprocessed_fif（递归查找）",
    )
    parser.add_argument("--full", action="store_true", help="打印每个文件的全部通道")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="精简模式：不逐个文件打印，只按目录汇总通道布局与一致性",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    fif_files = iter_fif_files(root)
    if not fif_files:
        print(f"未找到 FIF 文件: {root}（递归）")
        return

    print(f"扫描目录: {root}（递归）")
    print(f"共 {len(fif_files)} 个 FIF 文件\n")

    tables = {}
    for i, fif_file in enumerate(fif_files, 1):
        rel = fif_file.relative_to(root)
        kind, table = channel_table(fif_file)
        tables[str(rel)] = table
        if not args.summary:
            print(f"[{i}/{len(fif_files)}] {rel}")
            print_file(fif_file, kind, table, args.full)
            print()

    if args.summary:
        print_summary(tables)
        return

    # 按通道序列分组，检查一致性
    groups = {}
    for name, table in tables.items():
        groups.setdefault(tuple(table), []).append(name)

    print("=" * 70)
    print("通道顺序一致性")
    print("=" * 70)
    if len(groups) == 1:
        key = next(iter(groups))
        print(f"所有 {len(tables)} 个文件通道完全一致：共 {len(key)} 个通道，顺序相同。")
        return

    print(f"发现 {len(groups)} 种不同的通道布局：")
    for gi, (key, files) in enumerate(groups.items(), 1):
        print(f"\n布局 {gi}: n_chans={len(key)}，文件 {len(files)} 个")
        for name in files:
            print(f"    - {name}")

    base_key = str(fif_files[0].relative_to(root))
    base = tables[base_key]
    print(f"\n以 {base_key} 为基准，逐文件对比：")
    for name, table in tables.items():
        if name == base_key:
            continue
        print(f"  {name}: {first_diff(base, table)}")


if __name__ == "__main__":
    main()
