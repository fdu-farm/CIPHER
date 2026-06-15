import argparse
from pathlib import Path

import yaml

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from densenet_common import (
    HEAD_TYPE_SOFTMAX,
    TRANSFORM_IMAGENET,
    MultiHeadDenseNet,
    MultiHeadImageDataset,
    build_transforms,
    evaluate_one_epoch,
    format_head_metrics,
    label_dtype_for_head_type,
    log_epoch_metrics,
    seed_worker,
    set_backbone_trainable,
    set_seed,
    train_one_epoch,
)


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to config YAML.")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Train DenseNet-121 multi-head classifier with ImageNet initialization.",
        parents=[config_parser],
    )
    parser.add_argument("--train_csv", type=Path, default=None, help="Training CSV path")
    parser.add_argument("--val_csv", type=Path, default=None, help="Validation CSV path")
    parser.add_argument(
        "--image_root", type=Path, default=None, help="Optional root to prepend to CSV paths"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_grid", type=int, nargs="+", help="Batch sizes to search")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_grid", type=float, nargs="+", help="Learning rates to search")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Minimum LR for cosine annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross-entropy loss",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("best_model.pt"),
        help="Where to save the checkpoint with best validation loss",
    )
    parser.add_argument(
        "--search_report",
        type=Path,
        default=Path("best_search_result.txt"),
        help="Text file to record the best hyperparameter combo and its ACC/AUC/loss",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to resume training (expects model/optimizer state)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (applied to Python, NumPy, and PyTorch)",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("runs"),
        help="TensorBoard log directory (a run subdir is created per hyperparameter combo)",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Steps to accumulate gradients before optimizer step",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=0.0,
        help="Max gradient norm (0 disables clipping)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Freeze backbone for the first N epochs",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=None,
        help="Optional LR for backbone params (defaults to --lr)",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop if val loss does not improve for N epochs",
    )
    if config:
        known_args = {action.dest for action in parser._actions}
        filtered = {key: value for key, value in config.items() if key in known_args}
        for path_key in (
            "train_csv",
            "val_csv",
            "image_root",
            "save_path",
            "search_report",
            "resume_checkpoint",
            "log_dir",
        ):
            if path_key in filtered and filtered[path_key] is not None:
                filtered[path_key] = Path(filtered[path_key])
        parser.set_defaults(**filtered)
        unknown = sorted(set(config) - known_args)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")

    args = parser.parse_args()
    if args.train_csv is None or args.val_csv is None:
        parser.error("the following arguments are required: --train_csv, --val_csv")
    return args


def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers, seed):
    pin_memory = torch.cuda.is_available()
    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    return train_loader, val_loader


def train_single_run(
    train_loader,
    val_loader,
    args,
    device,
    batch_size: int,
    lr: float,
    use_grid: bool,
):
    run_name = f"bs{batch_size}_lr{lr:g}"
    print(f"\n=== Training with batch_size={batch_size}, lr={lr} ===")

    model = MultiHeadDenseNet(
        num_heads=train_loader.dataset.num_labels,
        head_type=HEAD_TYPE_SOFTMAX,
        use_imagenet=True,
    ).to(device)
    if args.backbone_lr is not None:
        optimizer = optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": args.backbone_lr},
                {"params": model.classifier_heads.parameters(), "lr": lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    writer = SummaryWriter(log_dir=args.log_dir / run_name)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    if args.resume_checkpoint and args.resume_checkpoint.exists():
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(
            checkpoint.get("model_state_dict", {}), strict=False
        )
        if missing:
            print(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading checkpoint: {unexpected}")
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resumed from checkpoint {args.resume_checkpoint} starting at epoch {start_epoch}")

    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    best_val_auc = float("-inf")
    best_epoch = None
    epochs_since_improve = 0

    combo_save_path = args.save_path
    if use_grid:
        combo_save_path = args.save_path.with_name(
            f"{args.save_path.stem}_bs{batch_size}_lr{lr:g}{args.save_path.suffix}"
        )

    try:
        if args.freeze_backbone_epochs > 0:
            set_backbone_trainable(model, False)
            print(f"Freezing backbone for first {args.freeze_backbone_epochs} epochs")

        for epoch in range(start_epoch, args.epochs + 1):
            if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
                set_backbone_trainable(model, True)
                print("Unfroze backbone for fine-tuning")

            (
                train_loss,
                train_acc,
                train_auc,
                train_head_acc,
                train_head_auc,
            ) = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                head_type=HEAD_TYPE_SOFTMAX,
                decision_threshold=0.5,
                grad_accum_steps=args.grad_accum_steps,
                grad_clip_norm=args.grad_clip_norm,
                scaler=scaler,
            )
            val_loss, val_acc, val_auc, val_head_acc, val_head_auc = evaluate_one_epoch(
                model,
                val_loader,
                criterion,
                device,
                head_type=HEAD_TYPE_SOFTMAX,
                decision_threshold=0.5,
            )
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_acc_str, train_auc_str = format_head_metrics(train_head_acc, train_head_auc)
            val_acc_str, val_auc_str = format_head_metrics(val_head_acc, val_head_auc)

            print(
                f"Epoch {epoch}: "
                f"Train Loss {train_loss:.4f} | Train ACC {train_acc:.4f} | Train AUC {train_auc:.4f} | "
                f"Val Loss {val_loss:.4f} | Val ACC {val_acc:.4f} | Val AUC {val_auc:.4f} | "
                f"LR {current_lr:.6f}\n"
                f"  Train per-head: {train_acc_str}; {train_auc_str}\n"
                f"  Val per-head: {val_acc_str}; {val_auc_str}"
            )

            log_epoch_metrics(
                writer,
                epoch,
                "train",
                train_loss,
                train_acc,
                train_auc,
                train_head_acc,
                train_head_auc,
            )
            log_epoch_metrics(
                writer,
                epoch,
                "val",
                val_loss,
                val_acc,
                val_auc,
                val_head_acc,
                val_head_auc,
            )
            writer.add_scalar("learning_rate", current_lr, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_since_improve = 0
                combo_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                        "batch_size": batch_size,
                        "lr": lr,
                    },
                    combo_save_path,
                )
                print(
                    f"Saved new best model for bs={batch_size}, lr={lr} with val loss {val_loss:.4f} "
                    f"to {combo_save_path}"
                )
            else:
                epochs_since_improve += 1
                if args.early_stop_patience and epochs_since_improve >= args.early_stop_patience:
                    print(
                        "Early stopping after epoch "
                        f"{epoch} (patience={args.early_stop_patience})"
                    )
                    break
    finally:
        writer.close()

    return best_val_loss, best_val_acc, best_val_auc, best_epoch, combo_save_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = build_transforms(TRANSFORM_IMAGENET, is_train=True)
    val_transform = build_transforms(TRANSFORM_IMAGENET, is_train=False)
    label_dtype = label_dtype_for_head_type(HEAD_TYPE_SOFTMAX)

    train_dataset = MultiHeadImageDataset(
        args.train_csv,
        args.image_root,
        train_transform,
        label_dtype=label_dtype,
    )
    val_dataset = MultiHeadImageDataset(
        args.val_csv,
        args.image_root,
        val_transform,
        label_dtype=label_dtype,
    )

    print(
        f"Detected {train_dataset.num_labels} label columns ->"
        f" {train_dataset.num_labels} classifier heads"
    )

    if train_dataset.num_labels != val_dataset.num_labels:
        raise ValueError(
            "Training and validation CSV files must have the same number of label columns"
        )

    lr_candidates = args.lr_grid if args.lr_grid else [args.lr]
    batch_candidates = args.batch_size_grid if args.batch_size_grid else [args.batch_size]
    use_grid = len(batch_candidates) > 1 or len(lr_candidates) > 1

    best_overall_loss = float("inf")
    best_overall_path: Path | None = None
    best_overall_acc = float("-inf")
    best_overall_auc = float("-inf")
    best_overall_combo: tuple[int, float, int] | None = None

    for batch_size in batch_candidates:
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size,
            args.num_workers,
            args.seed,
        )

        for lr in lr_candidates:
            (
                run_best_loss,
                run_best_acc,
                run_best_auc,
                run_best_epoch,
                run_best_path,
            ) = train_single_run(
                train_loader,
                val_loader,
                args,
                device,
                batch_size,
                lr,
                use_grid,
            )

            if run_best_loss < best_overall_loss:
                best_overall_loss = run_best_loss
                best_overall_acc = run_best_acc
                best_overall_auc = run_best_auc
                best_overall_path = run_best_path
                best_overall_combo = (batch_size, lr, run_best_epoch or 0)

    if best_overall_path:
        print(
            f"Best validation loss across all runs: {best_overall_loss:.4f} saved at {best_overall_path}"
        )

        bs, lr, epoch = best_overall_combo if best_overall_combo else (None, None, None)
        args.search_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.search_report, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "Best hyperparameter search result:",
                        "  backbone: DenseNet-121 (ImageNet initialized)",
                        f"  batch_size: {bs}",
                        f"  lr: {lr}",
                        f"  epoch: {epoch}",
                        f"  val_acc: {best_overall_acc:.4f}",
                        f"  val_auc: {best_overall_auc:.4f}",
                        f"  val_loss: {best_overall_loss:.4f}",
                        f"  checkpoint: {best_overall_path}",
                    ]
                )
            )
        print(f"Saved best search summary to {args.search_report}")


if __name__ == "__main__":
    main()
