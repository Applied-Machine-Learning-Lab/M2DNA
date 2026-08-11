#!/usr/bin/env python3
"""Train CGRclust's visual branch and export the best M2DNA-compatible checkpoint.

The script reuses the official CGRclust implementation without modifying that
repository. It runs the visual model five times by default, saves the best
epoch from every run, and copies the best run to one final checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CGRclust five times and export the best visual checkpoint for M2DNA."
    )
    parser.add_argument(
        "--cgrclust_dir",
        required=True,
        type=Path,
        help="Local clone of https://github.com/fatemehalipour/CGRclust",
    )
    parser.add_argument(
        "--dataset",
        default="01_Cypriniformes.fasta",
        type=str,
        help="FASTA file in CGRclust/data",
    )
    parser.add_argument("--k", default=6, type=int)
    parser.add_argument("--weak_mutation_rate", default=1e-4, type=float)
    parser.add_argument("--strong_mutation_rate", default=1e-2, type=float)
    parser.add_argument("--number_of_pairs", default=1, type=int)
    parser.add_argument("--number_of_models", default=5, type=int)
    parser.add_argument("--num_epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=7e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--temp_ins", default=0.1, type=float)
    parser.add_argument("--temp_clu", default=1.0, type=float)
    parser.add_argument("--embedding_dim", default=512, type=int)
    parser.add_argument("--feature_dim", default=128, type=int)
    parser.add_argument("--weight", default=0.7, type=float)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=Path,
        help="Output directory; defaults to M2DNA/checkpoints/<dataset-stem>",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    return parser.parse_args()


def import_cgrclust_modules(cgrclust_dir: Path):
    src_dir = (cgrclust_dir / "src").resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"CGRclust source directory not found: {src_dir}")

    # CGRclust uses absolute imports such as `from utils import ...`.
    sys.path.insert(0, str(src_dir))
    module_names = [
        "utils.data_setup",
        "utils.model",
        "utils.engine",
        "utils.loss_function",
        "utils.data_preprocess",
        "utils.augmentation_utils",
        "utils.utils",
    ]
    imported = {name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in module_names}
    return imported


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_run(
    modules,
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_loader: DataLoader,
    x_test,
    y_test,
    class_names,
    device: torch.device,
    seed: int,
    checkpoint_path: Path,
) -> float:
    model_mod = modules["model"]
    engine_mod = modules["engine"]
    loss_mod = modules["loss_function"]

    set_seed(seed)
    backbone = model_mod.BackBoneModel(input_shape=1, output_shape=args.embedding_dim)
    network = model_mod.Network(
        backbone=backbone,
        rep_dim=args.embedding_dim,
        feature_dim=args.feature_dim,
        class_num=len(class_names),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": network.backbone.parameters(), "lr": args.lr},
            {"params": network.instance_projector.parameters(), "lr": args.lr},
            {"params": network.cluster_projector.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    criterion_instance = loss_mod.InstanceLoss(args.batch_size, args.temp_ins, device).to(device)
    criterion_cluster = loss_mod.ClusterLoss(len(class_names), args.temp_clu, device).to(device)

    best_test_acc = -1.0
    best_epoch = -1
    for epoch in range(args.num_epochs):
        train_loss = engine_mod.train_step(
            model=network,
            dataloader=train_loader,
            criterion_instance=criterion_instance,
            criterion_cluster=criterion_cluster,
            optimizer=optimizer,
            scheduler=scheduler,
            weight=args.weight,
            device=device,
        )
        test_loss, test_acc = engine_mod.test_step(
            model=network,
            dataloader=test_loader,
            device=device,
        )

        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            best_epoch = epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": network.state_dict(),
                    "epoch": epoch,
                    "test_acc": float(test_acc),
                    "dataset": args.dataset,
                    "seed": seed,
                },
                checkpoint_path,
            )

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            print(
                f"[seed={seed}] epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} test_loss={test_loss:.6f} "
                f"test_acc={test_acc * 100:.4f}%"
            )

    # Re-evaluate the saved best epoch with CGRclust's global Hungarian metric.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model_state_dict"])
    _, _, _, final_acc = engine_mod.model_evaluation(network, X_test=x_test, y_test=y_test)
    print(
        f"[seed={seed}] best_epoch={best_epoch} "
        f"saved_test_acc={best_test_acc * 100:.4f}% "
        f"global_acc={final_acc * 100:.4f}%"
    )

    del network, optimizer, scheduler, criterion_instance, criterion_cluster
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(final_acc)


def main() -> None:
    args = parse_args()
    cgrclust_dir = args.cgrclust_dir.resolve()
    modules = import_cgrclust_modules(cgrclust_dir)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    device = torch.device(device_name)

    data_path = cgrclust_dir / "data" / args.dataset
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = args.output_dir or (
        Path(__file__).resolve().parents[1] / "checkpoints" / Path(args.dataset).stem
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"CGRclust: {cgrclust_dir}")
    print(f"Dataset: {data_path}")
    print(f"Output: {output_dir}")

    data_preprocess = modules["data_preprocess"]
    augmentation_utils = modules["augmentation_utils"]
    utils_mod = modules["utils"]
    data_setup = modules["data_setup"]

    records_df = data_preprocess.read_fasta(str(data_path))
    class_names = sorted(records_df.label.unique())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    # Match CGRclust's deterministic augmentation generation exactly.
    random.seed(42)
    np.random.seed(42)
    x_train, x_test, y_test = augmentation_utils.generate_pairs(
        data=records_df,
        class_to_idx=class_to_idx,
        k=args.k,
        number_of_pairs=args.number_of_pairs,
        mutation_rate_weak=args.weak_mutation_rate,
        mutation_rate_strong=args.strong_mutation_rate,
    )
    x_train, x_test = utils_mod.data_normalization(x_train, x_test)

    train_data = data_setup.PairSeqData(train_pairs=x_train, transform=None)
    test_data = data_setup.SeqData(
        sequences=x_test,
        labels=y_test,
        classes=class_names,
        class_to_idx=class_to_idx,
        transform=None,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    run_results = []
    for run_idx in range(args.number_of_models):
        seed = args.random_seed + run_idx
        run_checkpoint = output_dir / f"best_visual_model_{run_idx}.pth"
        print(f"\n{'=' * 20} CGRclust run {run_idx + 1}/{args.number_of_models} (seed={seed}) {'=' * 20}")
        acc = train_one_run(
            modules,
            args,
            train_loader,
            test_loader,
            x_test,
            y_test,
            class_names,
            device,
            seed,
            run_checkpoint,
        )
        run_results.append((acc, run_idx, run_checkpoint))

    best_acc, best_idx, best_path = max(run_results, key=lambda item: item[0])
    final_path = output_dir / "best_visual_model.pth"
    shutil.copy2(best_path, final_path)

    summary_path = output_dir / "selection_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        for acc, run_idx, path in run_results:
            handle.write(f"run={run_idx} accuracy={acc:.8f} checkpoint={path.name}\n")
        handle.write(f"selected_run={best_idx} selected_accuracy={best_acc:.8f}\n")

    print("\nSelection summary:")
    for acc, run_idx, path in run_results:
        print(f"  run {run_idx}: {acc * 100:.4f}% -> {path}")
    print(f"Selected run: {best_idx} ({best_acc * 100:.4f}%)")
    print(f"M2DNA checkpoint: {final_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
