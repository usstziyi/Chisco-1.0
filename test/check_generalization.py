"""诊断脚本：区分「纯记忆」与「eval / BatchNorm 管线 bug」。

背景：
    多轮训练都出现同一现象——train_top1 能到 1.000，val_top1 却停在随机
    （1/N），val_cos_pos ≈ 0。本脚本用已训好的 checkpoint 做两类判定。

第 1 部分（秒级，不需要重训）:
    a. 训练集 @ eval 模式     -> ≈1.0 说明模型「记住了」训练集，且 eval 管线正常
    b. 训练集 @ train-BN 模式 -> 与 a 接近说明 BN 没有 train/eval 落差
                                 （b 远高于 a 则说明 eval 模式被 BN 拖垮 = bug）
    c. 验证集 @ eval 模式     -> 若 ≈1/N 说明完全不泛化

第 2 部分（label-shuffle 对照，需要重训若干 epoch）:
    把训练集的文本标签整体打乱后重训。若 train_top1 仍能升到 ≈1.0，
    说明模型只是在「记 EEG→文本」的查表关系，而不是学到可迁移的 EEG 表征。

用法:
    # 自动选 outputs/ 下最新的 best.pt
    uv run python test/check_generalization.py

    # 指定 checkpoint
    uv run python test/check_generalization.py --ckpt outputs/<run>/best.pt

    # 只做第 1 部分（跳过重训）
    uv run python test/check_generalization.py --shuffle-epochs 0
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataloader import build_dataset, build_dataloaders, split_train_val  # noqa: E402
from encoder import (  # noqa: E402
    LaBSETextEncoder,
    TSConvFixedEEGEncoder,
    TSConvPointwiseEEGEncoder,
)

# 与训练脚本保持一致
EEG_SCALE = 1e6


# ============================================================
# 小工具
# ============================================================

def zscore_per_channel(eeg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """按通道 z-score（与训练脚本一致）：(B, C, T) 在时间维上标准化。"""
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False)
    return (eeg - mean) / (std + eps)


def dataset_texts(ds) -> list[str]:
    """取出任意 Dataset / Subset / ConcatDataset 的全量文本（不触碰 EEG）。"""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(ds, Subset):
        inner = dataset_texts(ds.dataset)
        return [inner[i] for i in ds.indices]
    if isinstance(ds, ConcatDataset):
        out: list[str] = []
        for d in ds.datasets:
            out.extend(dataset_texts(d))
        return out
    return [str(t) for t in ds.y]  # ChiscoEEGDataset


class ShuffledTextDataset(Dataset):
    """把 (eeg, text) 文本标签整体打乱后的 Dataset，用于 label-shuffle 对照。

    第 i 条 EEG 配的是打乱后的第 i 个文本；EEG 本身不动。
    """

    def __init__(self, base, indices, texts):
        self.base = base
        self.indices = list(indices)
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        sample = self.base[self.indices[i]]
        return {"eeg": sample["eeg"], "text": self.texts[i]}


def latest_checkpoint() -> Path:
    ckpts = sorted(
        (PROJECT_ROOT / "outputs").glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise FileNotFoundError("outputs/ 下没有找到任何 best.pt，请用 --ckpt 指定")
    return ckpts[-1]


# ============================================================
# 编码 / 指标
# ============================================================

@torch.no_grad()
def encode_all(eeg_encoder, loader, text_encoder, device, normalize):
    """对 loader 全量编码，返回 (eeg_emb (N,D), text_emb (N,D), texts)。"""
    eeg_embeddings = []
    texts: list[str] = []
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True) * EEG_SCALE
        if normalize:
            eeg = zscore_per_channel(eeg)
        eeg_embeddings.append(eeg_encoder(eeg).float().cpu())
        texts.extend(batch["text"])
    eeg_all = torch.cat(eeg_embeddings, dim=0)
    text_all = text_encoder.encode_unique(texts).float().cpu()
    return eeg_all, text_all, texts


def retrieval_metrics(eeg_all, text_all, topk: int = 5) -> dict:
    """全量 N×N 检索指标（与训练脚本 evaluate 口径一致）。"""
    similarity = eeg_all @ text_all.t()
    n = similarity.size(0)
    targets = torch.arange(n)
    k = min(topk, n)
    return {
        "n": n,
        "top1": similarity.argmax(dim=1).eq(targets).float().mean().item(),
        "topk": similarity.topk(k, dim=1).indices
        .eq(targets[:, None]).any(dim=1).float().mean().item(),
        "k": k,
        "cos_pos": similarity.diag().mean().item(),
        "chance": 1.0 / n,
    }


def build_encoder(ck, saved_args, text_dim, kind, device):
    kwargs = dict(
        n_chans=int(ck["n_chans"]),
        n_times=int(ck["n_times"]),
        k=40, m1=50, m2=100, s=20,
        projection_hidden_dim=512,
        embedding_dim=text_dim,
        drop_prob=float(saved_args.get("drop_prob", 0.5)),
    )
    if kind == "pointwise":
        encoder = TSConvPointwiseEEGEncoder(
            pointwise_channels=int(saved_args.get("pointwise_channels", 8)), **kwargs
        )
    else:
        encoder = TSConvFixedEEGEncoder(**kwargs)
    return encoder.to(device)


def report(tag: str, m: dict) -> None:
    print(
        f"  {tag:<22} N={m['n']:<6} top1={m['top1']:.4f} "
        f"top{m['k']}={m['topk']:.4f} cos+={m['cos_pos']:+.4f} "
        f"（随机基线 {m['chance']:.4f}）",
        flush=True,
    )


# ============================================================
# 第 2 部分：label-shuffle 对照
# ============================================================

def run_shuffle_control(base, train_idx, texts, saved_args, device, text_encoder,
                        kind, epochs, batch_size, seed):
    n = len(train_idx)
    perm = np.random.default_rng(seed + 1).permutation(n)
    n_fixed = int((perm == np.arange(n)).sum())
    shuffled_texts = [texts[perm[i]] for i in range(n)]

    print(
        f"\n[2] label-shuffle 对照：{n} 条训练样本，标签整体打乱"
        f"（其中 {n_fixed} 条恰好没变），重训 {epochs} 个 epoch\n"
    )

    ds = ShuffledTextDataset(base, train_idx, shuffled_texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = build_encoder(
        {"n_chans": base[0]["eeg"].shape[0], "n_times": base[0]["eeg"].shape[1]},
        saved_args, text_encoder.embedding_dim, kind, device,
    )
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=float(saved_args.get("lr", 1e-3)),
        weight_decay=float(saved_args.get("weight_decay", 1e-4)),
    )
    temperature = float(saved_args.get("temperature", 0.07))
    normalize = bool(saved_args.get("normalize", False))

    for epoch in range(1, epochs + 1):
        encoder.train()
        total_top1 = 0.0
        total_loss = 0.0
        seen = 0
        for batch in loader:
            text_emb = text_encoder.encode_unique(batch["text"])
            eeg = batch["eeg"].to(device, non_blocking=True) * EEG_SCALE
            if normalize:
                eeg = zscore_per_channel(eeg)
            eeg_emb = encoder(eeg)
            logits = eeg_emb @ text_emb.t() / temperature
            labels = torch.arange(eeg_emb.size(0), device=device)
            loss = 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = len(batch["text"])
            with torch.no_grad():
                top1 = (eeg_emb @ text_emb.t()).argmax(dim=1).eq(labels).float().mean().item()
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            seen += bs
        print(
            f"    shuffle epoch {epoch:2d}/{epochs} | "
            f"train loss {total_loss / seen:.4f} top1 {total_top1 / seen:.4f}",
            flush=True,
        )

    print(
        "\n    判读：若上面 train_top1 也能升到 ≈1.0，说明模型只是在『记』"
        "（打乱后照样能拟合），\n"
        "         即当前训练集上任何配对关系都能被记住，泛化失败不是管线 bug。"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="诊断：区分纯记忆与 eval/BN 管线 bug")
    parser.add_argument("--ckpt", default=None, help="best.pt 路径（默认取 outputs/ 下最新的）")
    parser.add_argument("--encoder", choices=["auto", "fixed", "pointwise"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256, help="诊断用 batch（全量 N×N，不影响指标）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--shuffle-epochs", type=int, default=10, help="label-shuffle 对照重训轮数（0 表示跳过）")
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.ckpt) if args.ckpt else latest_checkpoint()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = dict(ck.get("args", {}))

    kind = args.encoder
    if kind == "auto":
        kind = "pointwise" if any(k.startswith("pointwise_projection") for k in ck["model_state"]) else "fixed"

    print(
        f"checkpoint : {ckpt_path}\n"
        f"  encoder  : {kind}\n"
        f"  subject  : {saved_args.get('subject')} / {saved_args.get('task')} "
        f"/ days={saved_args.get('days')}\n"
        f"  normalize: {saved_args.get('normalize', False)}\n"
        f"  saved ep : {ck.get('epoch')}  val_top1={ck.get('val_top1'):.4f}\n"
    )

    # ---------------- 复现完全相同的 train/val 划分 ----------------
    subject = saved_args["subject"]
    task = saved_args["task"]
    days = saved_args["days"]
    val_ratio = float(saved_args.get("val_ratio", 0.2))
    seed = int(saved_args.get("seed", 42))

    train_loader, val_loader = build_dataloaders(
        subject=subject, task=task, days=days,
        val_ratio=val_ratio, batch_size=args.batch_size,
        num_workers=0, seed=seed, drop_last=False,
    )

    # ---------------- 模型 ----------------
    text_encoder = LaBSETextEncoder(
        model_name=saved_args.get("text_model", "sentence-transformers/LaBSE")
    ).to(device)
    encoder = build_encoder(ck, saved_args, text_encoder.embedding_dim, kind, device)
    encoder.load_state_dict(ck["model_state"])
    normalize = bool(saved_args.get("normalize", False))

    # ---------------- 第 1 部分 ----------------
    print("[1] 记忆 / 管线判定（全量 N×N 检索）")

    encoder.eval()
    tr_eeg, tr_txt, _ = encode_all(encoder, train_loader, text_encoder, device, normalize)
    report("训练集 @ eval", retrieval_metrics(tr_eeg, tr_txt, args.topk))

    # train-BN 模式（关掉 dropout，只留 BN 用 batch 统计），在副本上跑，避免污染 running stats
    probe = copy.deepcopy(encoder)
    probe.train()
    for module in probe.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    with torch.no_grad():
        probe_eeg, probe_txt, _ = encode_all(probe, train_loader, text_encoder, device, normalize)
    report("训练集 @ train-BN", retrieval_metrics(probe_eeg, probe_txt, args.topk))
    del probe

    encoder.eval()
    va_eeg, va_txt, _ = encode_all(encoder, val_loader, text_encoder, device, normalize)
    val_metrics = retrieval_metrics(va_eeg, va_txt, args.topk)
    report("验证集 @ eval", val_metrics)

    print(
        "\n  ---- 解读 ----\n"
        "  · 训练集 @ eval ≈ 1.0 且 训练集 @ train-BN 与之接近\n"
        "      -> eval 管线与 BN 都正常，模型确实「记住了」训练集。\n"
        "  · 验证集 @ eval 停在随机基线\n"
        "      -> 完全没有泛化：问题不在 eval / BN，而在表征或数据/任务本身。\n"
        "  · 若 训练集 @ eval 明显低于 训练集 @ train-BN\n"
        "      -> BN 存在 train/eval 落差，eval 模式被拖垮 = 需要修 BN。"
    )

    # ---------------- 第 2 部分 ----------------
    if args.shuffle_epochs > 0:
        base = build_dataset(subject, task, days)
        all_texts = dataset_texts(base)
        train_idx, _ = split_train_val(len(base), val_ratio=val_ratio, seed=seed)
        train_texts = [all_texts[i] for i in train_idx]
        run_shuffle_control(
            base, train_idx, train_texts, saved_args, device, text_encoder,
            kind, args.shuffle_epochs, int(saved_args.get("batch_size", 32)), seed,
        )


if __name__ == "__main__":
    main()
