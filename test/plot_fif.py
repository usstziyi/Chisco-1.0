"""
绘制预处理后 FIF 文件中 EEG 波形的简易脚本。

设计目标：
- 只读取并绘制指定的一段数据（由事件区间或时间区间决定），
  不会一次性加载 / 绘制整个文件。
- 同时兼容两种 FIF 内容：
    * Raw（连续数据）：直接按 [tmin, tmax] 裁剪绘制；
    * Epochs（分段数据）：用 events 把每一段还原到原始记录的绝对时间轴上再绘制，
      epoch 之间的空隙留空（不连接）。
- 通过形参控制：
    * n_channels                       绘制的 EEG 通道数量
    * start_event / end_event / occurrence   由事件描述确定绘制区间
    * tmin / tmax                      直接指定时间区间（秒，相对原始记录起点）

用法（命令行暴露 --fif / --epoch / --tmin / --tmax / --n-channels）：
    # --fif 直接当路径用（相对当前目录或绝对路径均可），不做被试目录解析
    uv run python plot_fif.py --fif preprocessed_fif/sub-05/eeg/sub-05_task-read_run-01_eeg.fif
    uv run python plot_fif.py --fif D:/abs/path/xxx.fif     # 也可直接给绝对路径
    uv run python plot_fif.py --fif preprocessed_fif/sub-05/eeg/sub-05_task-read_run-01_eeg.fif --tmin 300 --tmax 310
    uv run python plot_fif.py --fif preprocessed_fif/sub-05/eeg/sub-05_task-imagine_run-01_eeg.fif --epoch 19
    uv run python plot_fif.py --fif preprocessed_fif/sub-05/eeg/sub-05_task-read_run-01_eeg.fif --n-channels 20

绘图区间 / 通道数等由代码内部默认值决定：
    * n_channels = 10（可用 --n-channels 覆盖）
    * 未指定 tmin / tmax / 事件时，从第一条 annotation（或第一个 epoch）起绘制 DEFAULT_WINDOW 秒
    * 图片默认保存到脚本同级目录下的 test/plot/<文件名>_plot.png
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT_FOLDER = Path(__file__).resolve().parent
METHOD_STR = "prep"
PREP_ROOT = ROOT_FOLDER / "preprocess_output" / METHOD_STR

# 未指定事件 / 时间区间时，默认绘制多长的一段
DEFAULT_WINDOW = 10.0


def plot_folder():
    """图片默认保存目录：脚本同级目录下的 plot/（即 test/plot）。"""
    return ROOT_FOLDER / "plot"


def _fif_kind(fif_file):
    """判断 FIF 里装的是连续 Raw 还是分段 Epochs。"""
    from mne._fiff.constants import FIFF
    from mne._fiff.open import fiff_open
    from mne._fiff.tree import dir_tree_find

    ff, tree, _ = fiff_open(Path(fif_file))
    try:
        raw_blocks = (
            FIFF.FIFFB_RAW_DATA,
            FIFF.FIFFB_CONTINUOUS_DATA,
            FIFF.FIFFB_IAS_RAW_DATA,
        )
        if any(dir_tree_find(tree, block) for block in raw_blocks):
            return "raw"
        if dir_tree_find(tree, FIFF.FIFFB_MNE_EPOCHS):
            return "epochs"
    finally:
        ff.close()
    raise ValueError(f"无法识别的 FIF 内容（既不是 Raw 也不是 Epochs）：{fif_file}")


def read_any_fif(fif_file):
    """读入 FIF，返回 (inst, kind)，kind 为 "raw" 或 "epochs"。

    预处理产物既可能是连续 Raw，也可能是已经分段的 Epochs，这里自动识别。
    """
    kind = _fif_kind(fif_file)
    if kind == "raw":
        return mne.io.read_raw_fif(fif_file, preload=False, verbose=False), "raw"
    # 文件名不符合 MNE 的 *_epo.fif 约定，这里忽略该命名警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mne.read_epochs(fif_file, preload=False, verbose=False), "epochs"


def resolve_window(
    onsets,
    descriptions,
    duration,
    tmin=None,
    tmax=None,
    start_event=None,
    end_event=None,
    occurrence=0,
    default_duration=DEFAULT_WINDOW,
):
    """确定要绘制的 [tmin, tmax] 时间区间（秒，相对记录起点）。

    优先级：
    1. 显式给定 tmin / tmax 时直接使用（只给 tmin 时右界为 tmin + default_duration）。
    2. 给定 start_event 时，取该描述第 occurrence 次出现作为左界；
       若同时给定 end_event，右界为该左界之后第一次 end_event 的 onset，
       否则右界为左界 + default_duration。
    3. 都没给时，取第一条事件作为起点。
    """
    if tmin is not None or tmax is not None:
        left = 0.0 if tmin is None else float(tmin)
        right = left + default_duration if tmax is None else float(tmax)
        return left, right

    onsets = np.asarray(onsets, dtype=float)
    descriptions = list(descriptions)

    if start_event is None:
        if len(onsets) == 0:
            return 0.0, min(default_duration, duration)
        left = float(onsets[0])
    else:
        matches = [i for i, desc in enumerate(descriptions) if desc == start_event]
        if not matches:
            raise ValueError(
                f"事件描述 {start_event!r} 不存在，可选：{sorted(set(descriptions))}"
            )
        if not 0 <= occurrence < len(matches):
            raise ValueError(
                f"{start_event!r} 共出现 {len(matches)} 次，occurrence={occurrence} 越界。"
            )
        left = float(onsets[matches[occurrence]])

    if end_event is None:
        right = left + default_duration
    else:
        later = [
            onset
            for onset, desc in zip(onsets, descriptions)
            if desc == end_event and onset > left
        ]
        if not later:
            raise ValueError(
                f"{left:.3f}s 之后没有找到 {end_event!r}，"
                f"可选：{sorted(set(descriptions))}"
            )
        right = float(min(later))

    return left, right


def _epochs_window(epochs, channel_indices, onsets, left, right, sfreq):
    """把落在 [left, right] 内的 epoch 段铺到一条绝对时间轴上（空隙留 NaN）。

    返回 (times, data)，data 单位 µV，shape=(n_channels, n_times)。
    """
    n_samples = max(int(round((right - left) * sfreq)), 1)
    times = left + np.arange(n_samples) / sfreq
    data = np.full((len(channel_indices), n_samples), np.nan)

    seg_len = float(epochs.tmax - epochs.tmin)
    ends = onsets + seg_len
    selected = np.where((ends > left) & (onsets < right))[0]
    if len(selected) == 0:
        raise ValueError(
            f"时间区间 [{left:.3f}, {right:.3f}] s 内没有任何 epoch 数据。"
        )

    local_offsets = np.arange(epochs.times.size) / sfreq
    with mne.use_log_level("WARNING"):
        for i in selected:
            seg = epochs[i].get_data()[0]  # 只加载这一段：(n_ch_all, T)
            seg_times = onsets[i] + local_offsets
            idx = np.round((seg_times - left) * sfreq).astype(int)
            ok = (idx >= 0) & (idx < n_samples)
            data[:, idx[ok]] = seg[channel_indices][:, ok] * 1e6  # V -> µV

    return times, data


def plot_eeg_window(
    fif_file,
    n_channels=10,
    tmin=None,
    tmax=None,
    start_event=None,
    end_event=None,
    occurrence=0,
    epoch=None,
    channels=None,
    output=None,
    show=False,
):
    """绘制一段 EEG 波形。

    Parameters
    ----------
    fif_file : str | Path
        预处理后的 FIF 文件（Raw 或 Epochs 均可）。
    n_channels : int
        绘制多少个 EEG 通道（从文件里的 EEG 通道中按顺序取前 n 个）。
    tmin, tmax : float | None
        直接指定的时间区间（秒，相对原始记录起点），优先级高于事件区间。
    start_event, end_event : str | None
        用事件描述确定绘制区间，见 resolve_window。
    occurrence : int
        start_event 取第几次出现（从 0 开始）。
    epoch : int | None
        直接指定要绘制的第几个 epoch（从 0 开始，仅对 Epochs 文件有效）；
        给定时绘制该 epoch 的完整区间，优先级高于 tmin / tmax。
    channels : list[str] | None
        显式指定要绘制的通道名；给定时忽略 n_channels。
    output : str | Path | None
        图片保存路径（仅在 show=False 时生效）；
        默认保存到 plot 目录下 <文件名>_plot.png。
    show : bool
        为 True 时只弹出交互窗口、不保存图片；
        为 False 时才保存图片到 output，不弹窗。
    """
    fif_file = Path(fif_file)
    inst, kind = read_any_fif(fif_file)

    eeg_names = [
        name
        for name, ch_type in zip(inst.ch_names, inst.get_channel_types())

        if ch_type == "eeg"
    ]


    selected = list(channels) if channels is not None else eeg_names[:n_channels]
    if not selected:
        raise ValueError("没有可绘制的 EEG 通道。")

    if kind == "raw":
        onsets = np.asarray(inst.annotations.onset, dtype=float)
        descriptions = list(inst.annotations.description)
        duration = float(inst.times[-1])
    else:
        sfreq = float(inst.info["sfreq"])
        # 每个 epoch 段在原始记录里的绝对起点 = 事件位置 + tmin
        onsets = inst.events[:, 0].astype(float) / sfreq + float(inst.tmin)
        descriptions = [f"ep{i}" for i in range(len(inst))]
        duration = float(onsets.max() + (inst.tmax - inst.tmin))

    if epoch is not None:
        if kind != "epochs":
            raise ValueError("--epoch 只能用于 Epochs 文件，当前文件是连续 Raw。")
        if not 0 <= epoch < len(inst):
            raise ValueError(
                f"epoch 索引 {epoch} 越界，文件共有 {len(inst)} 个 epoch（0..{len(inst) - 1}）。"
            )
        left = float(onsets[epoch])
        right = left + float(inst.tmax - inst.tmin)
    else:
        left, right = resolve_window(
            onsets, descriptions, duration, tmin, tmax, start_event, end_event, occurrence
        )
    left = max(left, 0.0)
    right = min(right, duration)
    if right <= left:
        raise ValueError(f"无效的绘制区间：[{left}, {right}]")

    # 窗口内的事件（绝对时间），crop / 拼接后 x 轴仍是绝对时间
    markers = [
        (float(onset), desc)
        for onset, desc in zip(onsets, descriptions)
        if left <= onset <= right
    ]

    if kind == "raw":
        raw = inst
        raw.pick(selected)
        raw.crop(tmin=left, tmax=right)
        raw.load_data()
        data = raw.get_data() * 1e6  # V -> µV
        # crop 后 raw.times 从 0 重新计时，还原成相对记录起点的绝对时间
        times = raw.times + left
        ch_names = list(raw.ch_names)
    else:
        ch_idx = [inst.ch_names.index(name) for name in selected]
        times, data = _epochs_window(inst, ch_idx, onsets, left, right, sfreq)
        ch_names = selected

    n_plot = len(ch_names)
    peak = float(np.nanmax(np.abs(data))) if data.size else 1.0
    if not np.isfinite(peak) or peak == 0.0:
        peak = 1.0
    separation = max(peak * 2.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.45 * n_plot)))
    for i, ch_name in enumerate(ch_names):
        ax.plot(times, data[i] + i * separation, lw=0.6, color="black")

    ax.set_yticks(np.arange(n_plot) * separation)
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlim(left, right)
    ax.set_xlabel("Time (s)")
    scope = f"epoch {epoch}/{len(inst) - 1} | " if epoch is not None else ""
    ax.set_title(
        f"{fif_file.stem}\n"
        f"{scope}{left:.2f} - {right:.2f} s | {n_plot} EEG channels ({kind})"
    )

    for onset, desc in markers:
        ax.axvline(onset, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.text(
            onset,
            n_plot * separation,
            desc,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color="red",
        )

    fig.tight_layout()

    # show=True：只弹窗查看，不落盘
    if show:
        plt.show()
        return None

    # show=False：保存图片，不弹窗
    if output is None:
        suffix = f"_ep{epoch:03d}" if epoch is not None else ""
        output = plot_folder() / f"{fif_file.stem}{suffix}_plot.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="绘制预处理 FIF 中一段 EEG 波形（不会一次性绘制整个数据）。"
    )
    parser.add_argument(
        "--fif",
        required=True,
        help="fif 文件路径（相对当前目录或绝对路径均可），"
        "如 preprocessed_fif/sub-05/eeg/sub-05_task-read_run-01_eeg.fif",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="直接绘制第几个 epoch（从 0 开始，仅对 Epochs 文件有效）",
    )
    parser.add_argument("--tmin", type=float, default=None, help="起始时间（秒）")
    parser.add_argument("--tmax", type=float, default=None, help="结束时间（秒）")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=10,
        help="绘制的 EEG 通道数量（从 EEG 通道中按顺序取前 n 个，默认 10）",
    )
    args = parser.parse_args()

    if args.epoch is not None and (args.tmin is not None or args.tmax is not None):
        parser.error("--epoch 与 --tmin/--tmax 不能同时使用。")

    plot_eeg_window(
        fif_file=args.fif,
        epoch=args.epoch,
        tmin=args.tmin,
        tmax=args.tmax,
        n_channels=args.n_channels,
    )


if __name__ == "__main__":
    main()
