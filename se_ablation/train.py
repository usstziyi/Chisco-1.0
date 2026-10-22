"""SE 消融的对比学习训练循环（单被试 + 单 task），复制自 ``baseline/train.py``。

与 baseline 的唯一差异是模型侧多了一个 ``--se-mode`` 开关：通道维 SE 插在
``front`` / ``back`` / ``mid`` 三个位置之一，``none`` 与 baseline 完全等价。
数据、采样、损失、评测、日志格式全部沿用 baseline，方便四种配置直接对比。

与 ``temp/baseline/train.py`` 同构，差异来自本数据集的数据组织：

- 一次只跑一个 ``(subject, task)``，对应一个 ``ChiscoEEGDataset``；
- 样本是**整句** EEG ``(122, T)``，不是字级 token，所以模型也不需要 mask；
- CSV 已由 ``scripts/make_text_csv.py`` 按句子去重（每句只保留首次出现的试次），
  所以 batch 内句子两两不同，损失用标准对称 InfoNCE：对角线即正样本。

数据
----
句子来自 ``datasets/preprocessed/{subject}_{task}.csv``，EEG 来自
``datasets/ds005170-download/derivatives/preprocessed_pkl`` 下的 pkl。
train / val 按 **句子** 切分（不是按样本），保证两侧句子集合完全不相交。

数据加载（本机特有的约束）
--------------------------
``ChiscoEEGDataset`` 每访问一个 run 就要完整 unpickle 一个 465 MB 的 pkl，再用
LRU 缓存（默认只留 2 个 run）；45 个 run 解包态合计约 21 GB，本机内存装不下。
若直接 ``shuffle=True``，缓存命中率会掉到几个百分点，每个 batch 都要反复重载
几十个文件，完全跑不动。

所以这里用 ``RunBatchSampler``：把样本按 run 分组后再切 batch，每个 epoch 只打乱
**run 的顺序**和 **run 内样本的顺序**，使每个 run 的样本在遍历流里始终是连续的一块，
每个 run 每轮只被加载一次（命中 LRU 缓存）。代价是一个 batch 基本只来自单个 run，
只在 run 边界处横跨相邻的几个 run（实际数据里 run 长度远大于 batch_size，通常只跨 2 个）；
并且 ``num_workers`` 必须为 0，否则每个 worker 都有自己的缓存副本。

损失
----
标准对称 InfoNCE：

    logits = eeg_embedding @ text_embedding.T / temperature
    loss   = 0.5 * (cross_entropy(logits, arange(B))
                  + cross_entropy(logits.T, arange(B)))

batch 内第 i 条 EEG 与第 i 条文本互为正样本，其余全部视为负样本。
温度系数默认 0.2；同 batch 内的其他句子构成 in-batch 负样本。

评测
----
- ``eeg2text``：候选 = val 里的唯一句子，命中 = 前 k 名。
- ``text2eeg``：候选 = 全部 val EEG 样本（每句一条），命中 = 前 k 名里出现自身。

``val_loss`` 越小越好，其余指标越大越好；best checkpoint 默认按 ``val_loss`` 选。

文本侧用冻结的 LaBSE，所有句子的文本 embedding 在训练前一次性算好、做成查表，
训练时只做索引。

用法::

    uv run python se_ablation/train.py --se-mode mid
    uv run python se_ablation/train.py --task imagine --epochs 20
    uv run python se_ablation/train.py --se-mode front --max-runs 5   # 冒烟
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

# 支持 `python se_ablation/train.py` 直接执行（此时 sys.path 不含项目根目录）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from se_ablation.dataset import ChiscoEEGDataset
from se_ablation.text_encoder import LaBSETextEncoder
# 模型是 SE 消融版；dataset / text_encoder 为 baseline 的同名拷贝，实验目录自包含
from se_ablation.model import ConvTimeAttentionEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "se_ablation"

# train_epochs.csv 每轮追加一行，字段顺序即表头顺序
EPOCH_LOG_FIELDS = [
    "epoch", "lr", "train_loss",
    "val_loss", "n_val", "n_unique_sentences", "n_text_candidates",
    "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5",
    "is_best", "temperature",
]

# train_steps.csv 每个训练 step 追加一行：第 0 列 epoch、第 1 列 step
TRAIN_STEP_FIELDS = ["epoch", "step", "loss", "lr", "batch_size"]


class RunBatchSampler(Sampler[list[int]]):
    """按 run 分组的 batch 采样器，见模块文档「数据加载」一节。

    产出的每个 batch 是一串**全局**样本下标（相对底层 dataset），所以在
    ``DataLoader`` 里必须把完整的 dataset 传进去，不能再套 ``Subset``。

    Args:
        dataset: ``ChiscoEEGDataset``，只用来读 ``rows`` 里的 run 编号。
        indices: 本次要遍历的全局样本下标（例如 train / val 各自的那一半）。
        batch_size: 每个 batch 的样本数。
        shuffle: 是否打乱 run 顺序与 run 内样本顺序；验证集传 False。
        drop_last: 是否丢掉最后一个不满的 batch。
        seed: 打乱用的种子；实际种子是 ``seed + epoch``，保证每个 epoch 不同。
    """

    def __init__(
        self,
        dataset: ChiscoEEGDataset,
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
            # 若干个 run；produced 的下标流里每个 run 仍是连续的一块，不会回头访问
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
    dataset: ChiscoEEGDataset, indices: Sequence[int], val_ratio: float, seed: int
):
    """按句子切分 train / val，返回 ``(train_idx, val_idx, 统计信息)``。

    以句子为单位而非样本：CSV 已按句子去重，当前与按样本切分等价；保留句子级
    切分是为了防止后续数据重新出现同句多试次时，同一句话同时落在两侧造成泄漏。
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


def build_text_lookup(encoder, sentences, device, chunk: int = 64):
    """把所有唯一句子编码成 embedding，返回 ``({sentence: (D,)}, embedding_dim)``。"""
    unique = sorted(set(sentences))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), chunk):
            vectors.append(encoder(unique[start : start + chunk]))
    table = torch.cat(vectors, dim=0).to(device)
    embedding_dim = table.shape[1]
    return dict(zip(unique, table)), embedding_dim


def gather_text(sentences, lookup, device) -> torch.Tensor:
    """按句子取回文本 embedding，堆成 (B, D)。"""
    return torch.stack([lookup[s] for s in sentences]).to(device)


def symmetric_infonce(
    eeg: torch.Tensor, text: torch.Tensor, temperature: float
) -> torch.Tensor:
    """标准对称 InfoNCE：EEG->文本 与 文本->EEG 两个方向取平均。

    第 i 条 EEG 与第 i 条文本互为正样本，batch 内其余样本全部为负样本。
    两侧都已 L2 归一化，点积即余弦相似度。
    """
    logits = eeg @ text.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def train_one_epoch(
    model, loader, sampler, lookup, optimizer, device, temperature,
    epoch, step_log_path=None,
):
    """跑一个 epoch，返回平均 loss。

    传了 ``step_log_path`` 就把每个 step 追加到 CSV：第 0 列 epoch、第 1 列 step。
    """
    model.train()
    sampler.set_epoch(epoch)
    total_loss, total_n = 0.0, 0
    lr = optimizer.param_groups[0]["lr"]
    for step, (eeg, sentences) in enumerate(loader, start=1):
        eeg = eeg.to(device)                              # (B, C, T)
        text = gather_text(sentences, lookup, device)     # (B, D)

        embedding = model(eeg)
        loss = symmetric_infonce(embedding, text, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = len(sentences)
        if step_log_path is not None:
            append_csv_row(step_log_path, TRAIN_STEP_FIELDS, {
                "epoch": epoch,
                "step": step,
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr:.6f}",
                "batch_size": batch,
            })

        total_loss += loss.item() * batch
        total_n += batch
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, sampler, lookup, device, temperature) -> dict:
    """在整个 val 集上算双向检索指标与全库对比损失。"""
    model.eval()
    sampler.set_epoch(0)  # 验证集不打乱，固定顺序便于复现

    eeg_parts, sentences = [], []
    for eeg, batch_sentences in loader:
        eeg_parts.append(model(eeg.to(device)))
        sentences.extend(batch_sentences)

    eeg = torch.cat(eeg_parts, dim=0)
    text = gather_text(sentences, lookup, device)
    n = eeg.size(0)

    metrics = {
        "val_loss": float(symmetric_infonce(eeg, text, temperature)),
        "n_val": n,
        "n_unique_sentences": len(set(sentences)),
    }

    # ---- eeg -> text：候选池 = val 里的唯一句子 ----
    unique_sentences = sorted(set(sentences))
    sentence_to_col = {s: i for i, s in enumerate(unique_sentences)}
    text_unique = gather_text(unique_sentences, lookup, device)
    target = torch.tensor(
        [sentence_to_col[s] for s in sentences], device=device, dtype=torch.long
    )  # 每条样本的正确答案列号
    ranked = (eeg @ text_unique.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(target[:, None])
    metrics["n_text_candidates"] = len(unique_sentences)
    metrics["eeg2text_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["eeg2text_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    # ---- text -> eeg：候选池 = 全部 val EEG 样本（每句一条），命中 = 前 k 名里出现自身 ----
    ranked = (text @ eeg.t()).argsort(dim=1, descending=True)
    hit = ranked.eq(torch.arange(n, device=device)[:, None])
    metrics["text2eeg_top1"] = float(hit[:, :1].any(dim=1).float().mean())
    metrics["text2eeg_top5"] = float(hit[:, :5].any(dim=1).float().mean())

    return metrics


def build_loader(
    dataset: ChiscoEEGDataset,
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
    parser = argparse.ArgumentParser(description="SE 消融：单被试单 task 的 EEG–文本对比学习。")
    parser.add_argument("--subject", default="sub-01", help="被试编号，如 sub-01（默认 sub-01）")
    parser.add_argument("--task", default="imagine", choices=["reading", "imagine"], help="reading / imagine（默认 imagine）")
    parser.add_argument("--se-mode", default="none", choices=["none", "front", "back", "mid"], help="通道维 SE 的位置：none / front（输入前置）/ back（特征后置）/ mid（空间卷积前，默认 none）")
    parser.add_argument("--se-reduction", type=int, default=4, help="SE 压缩比 r（默认 4）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集句子比例（默认 0.2）")
    parser.add_argument("--batch-size", type=int, default=64, help="batch 大小（默认 64）")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数（默认 30）")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率（默认 1e-3）")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW 权重衰减（默认 1e-3）")
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"], help="学习率调度：cosine 按 epoch 余弦衰减到 0（默认），none 恒定")
    parser.add_argument("--select-by", default="val_loss", choices=["val_loss", "eeg2text_top1", "eeg2text_top5", "text2eeg_top1", "text2eeg_top5"], help="用哪个指标选 best checkpoint（默认 val_loss，越小越好；其余越大越好）")
    parser.add_argument("--temperature", type=float, default=0.2, help="对比学习温度系数（默认 0.2）")
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
          f"select={args.select_by} temperature={args.temperature} seed={args.seed}")

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_args_json(args, out_dir)

    # ---------------- 数据 ----------------
    dataset = ChiscoEEGDataset(args.subject, args.task)
    first_eeg, _ = dataset[0]  # 触发一次读取，确定通道数 / 时间长度
    n_chans, n_samples = first_eeg.shape

    indices = list(range(len(dataset)))
    if args.max_runs > 0:
        keep_runs = sorted({int(row["run"]) for row in dataset.rows})[: args.max_runs]
        keep = set(keep_runs)
        indices = [i for i in indices if int(dataset.rows[i]["run"]) in keep]

    train_idx, val_idx, info = split_by_sentence(
        dataset, indices, args.val_ratio, args.seed
    )
    sentences = [dataset.rows[i]["sentence"] for i in indices]
    print(
        f"[Data] {args.subject} {args.task} 样本 {len(dataset)}"
        f"（使用 {len(indices)} 条 / {len({int(dataset.rows[i]['run']) for i in indices})} 个 run）\n"
        f"       每个 run 只加载一次；n_chans={n_chans} T={n_samples}\n"
        f"       句子 {len(set(sentences))} 条 -> "
        f"train {info['n_unique_train']} / val {info['n_unique_val']} "
        f"(样本 {len(train_idx)} / {len(val_idx)})，句子重叠={info['sentence_overlap']}"
    )

    train_loader, train_sampler = build_loader(
        dataset, train_idx, args.batch_size, True, args.seed, args.num_workers
    )
    val_loader, val_sampler = build_loader(
        dataset, val_idx, args.batch_size, False, args.seed, args.num_workers
    )

    # ---------------- 文本 embedding（冻结，一次性算好） ----------------
    print("[Text] 加载 LaBSE（冻结） ...")
    text_encoder = LaBSETextEncoder()
    lookup, embedding_dim = build_text_lookup(text_encoder, sentences, device)
    print(f"[Text] 已缓存 {len(lookup)} 条句子，embedding_dim={embedding_dim}")

    # ---------------- 模型 ----------------
    model = ConvTimeAttentionEncoder(
        n_samples=n_samples,
        n_chans=n_chans,
        embedding_dim=embedding_dim,
        se_mode=args.se_mode,
        se_reduction=args.se_reduction,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Model] {type(model).__name__} 参数量={n_params:,} "
        f"d_model={model.d_model} k={model.k} T2={model.n_time} "
        f"se_mode={model.se_mode} r={args.se_reduction}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # 恒定学习率在 val_loss 走平后仍会继续拟合，余弦衰减到 0 有助于收敛
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    # ---------------- 训练 ----------------
    # 文件名带上 se_mode，便于把多次消融的文件并排对比
    tag = f"{args.subject}_{args.task}_se-{args.se_mode}"
    ckpt_path = out_dir / f"{tag}_best.pt"
    train_epochs_path = out_dir / f"{tag}_train_epochs.csv"
    train_steps_path = out_dir / f"{tag}_train_steps.csv"

    lower_is_better = args.select_by == "val_loss"
    best_score = float("inf") if lower_is_better else -float("inf")
    best_metrics = None
    header = (
        f"{'epoch':>5} {'lr':>8} {'train_loss':>10} {'val_loss':>9} "
        f"{'e2t@1':>7} {'e2t@5':>7} {'t2e@1':>7} {'t2e@5':>7}"
    )
    print(f"\n{header}  (best by {args.select_by})")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, train_sampler, lookup, optimizer, device,
            args.temperature, epoch=epoch, step_log_path=train_steps_path,
        )
        if scheduler is not None:
            scheduler.step()
        metrics = evaluate(
            model, val_loader, val_sampler, lookup, device, args.temperature
        )

        score = metrics[args.select_by]
        improved = score < best_score if lower_is_better else score > best_score
        flag = ""
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            flag = " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "n_chans": n_chans,
                    "n_samples": n_samples,
                    "embedding_dim": embedding_dim,
                    "metrics": metrics,
                },
                ckpt_path,
            )

        print(
            f"{epoch:>5} {epoch_lr:>8.2e} {train_loss:>10.4f} "
            f"{metrics['val_loss']:>9.4f} "
            f"{metrics['eeg2text_top1']:>7.4f} {metrics['eeg2text_top5']:>7.4f} "
            f"{metrics['text2eeg_top1']:>7.4f} {metrics['text2eeg_top5']:>7.4f}{flag}"
        )

        # 本轮指标追加到 train_epochs.csv（首轮自动写表头）
        append_csv_row(train_epochs_path, EPOCH_LOG_FIELDS, {
            "epoch": epoch,
            "lr": f"{epoch_lr:.6f}",
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{metrics['val_loss']:.6f}",
            "n_val": metrics["n_val"],
            "n_unique_sentences": metrics["n_unique_sentences"],
            "n_text_candidates": metrics["n_text_candidates"],
            "eeg2text_top1": f"{metrics['eeg2text_top1']:.6f}",
            "eeg2text_top5": f"{metrics['eeg2text_top5']:.6f}",
            "text2eeg_top1": f"{metrics['text2eeg_top1']:.6f}",
            "text2eeg_top5": f"{metrics['text2eeg_top5']:.6f}",
            "is_best": int(improved),
            "temperature": args.temperature,
        })

    # 随机基线也要按两个方向的候选池分别算
    n_val = best_metrics["n_val"]
    n_cand = best_metrics["n_text_candidates"]
    print(
        f"\n[Done] best {args.select_by} = {best_score:.4f}\n"
        f"       eeg2text 候选 {n_cand} 句，随机基线 top-1={1 / n_cand:.4f} "
        f"top-5={min(5, n_cand) / n_cand:.4f}\n"
        f"       text2eeg 候选 {n_val} 条，随机基线 top-1={1 / n_val:.4f} "
        f"top-5={min(5, n_val) / n_val:.4f}\n"
        f"       checkpoint: {ckpt_path}\n"
        f"       epochs:     {train_epochs_path}\n"
        f"       steps:      {train_steps_path}"
    )


if __name__ == "__main__":
    main()
