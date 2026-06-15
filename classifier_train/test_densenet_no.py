import argparse
from pathlib import Path

import yaml

import torch
from torch.utils.data import DataLoader

from densenet_common import (
    HEAD_TYPE_SIGMOID,
    TRANSFORM_MEDICAL,
    MultiHeadDenseNet,
    MultiHeadImageDataset,
    build_transforms,
    evaluate_metrics,
    format_head_metrics,
    label_dtype_for_head_type,
    load_checkpoint,
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
        description="Evaluate DenseNet-121 multi-head classifier from CSV labels.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        nargs="+",
        help="One or more CSV files for evaluation",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to the checkpoint (e.g., best_model.pt)",
    )
    parser.add_argument(
        "--image_root", type=Path, default=None, help="Optional root to prepend to CSV paths"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--imagenet_init",
        action="store_true",
        help="Initialize DenseNet with ImageNet weights before loading checkpoint",
    )
    parser.add_argument("--imagenet", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional path to write evaluation metrics as text",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.3,
        help="Sigmoid threshold for reporting accuracy",
    )
    if config:
        known_args = {action.dest for action in parser._actions}
        filtered = {key: value for key, value in config.items() if key in known_args}
        if "csv" in filtered and filtered["csv"] is not None:
            csv_value = filtered["csv"]
            if isinstance(csv_value, (str, Path)):
                filtered["csv"] = [Path(csv_value)]
            else:
                filtered["csv"] = [Path(item) for item in csv_value]
        for path_key in ("weights", "image_root", "report_path"):
            if path_key in filtered and filtered[path_key] is not None:
                filtered[path_key] = Path(filtered[path_key])
        parser.set_defaults(**filtered)
        unknown = sorted(set(config) - known_args)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")

    args = parser.parse_args()
    if args.csv is None or args.weights is None:
        parser.error("the following arguments are required: --csv, --weights")
    args.csv = [Path(item) for item in args.csv]
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_transforms(TRANSFORM_MEDICAL, is_train=False)
    label_dtype = label_dtype_for_head_type(HEAD_TYPE_SIGMOID)
    datasets = []
    for csv_path in args.csv:
        ds = MultiHeadImageDataset(
            csv_path,
            args.image_root,
            transform,
            label_dtype=label_dtype,
        )
        print(
            f"Detected {ds.num_labels} label columns in {csv_path} -> {ds.num_labels} classifier heads"
        )
        datasets.append(ds)
    num_labels_set = {ds.num_labels for ds in datasets}
    if len(num_labels_set) != 1:
        raise ValueError(
            f"All CSV files must have the same number of label columns; got {sorted(num_labels_set)}"
        )

    imagenet_init = args.imagenet_init or args.imagenet
    model = MultiHeadDenseNet(
        num_heads=datasets[0].num_labels,
        head_type=HEAD_TYPE_SIGMOID,
        use_imagenet=imagenet_init,
    ).to(device)

    checkpoint, load_result = load_checkpoint(model, args.weights, device)
    if load_result.unexpected_keys:
        print(
            "Warning: Dropped unexpected checkpoint keys: "
            + ", ".join(sorted(load_result.unexpected_keys))
        )
    if load_result.missing_keys:
        print(
            "Warning: Missing checkpoint keys (initialized randomly): "
            + ", ".join(sorted(load_result.missing_keys))
        )
    print(
        f"Loaded checkpoint from {args.weights} (epoch={checkpoint.get('epoch', 'N/A')}, val_auc={checkpoint.get('val_auc', 'N/A')})"
    )

    report_lines = []
    header = (
        f"weights={args.weights}",
        f"imagenet_init={imagenet_init}",
        f"decision_threshold={args.decision_threshold}",
        f"epoch={checkpoint.get('epoch', 'N/A')}",
        f"val_auc={checkpoint.get('val_auc', 'N/A')}",
        "backbone=DenseNet-121",
    )
    report_lines.append("Evaluation summary: " + ", ".join(header))

    for csv_path, dataset in zip(args.csv, datasets):
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        avg_acc, avg_auc, per_head_acc, per_head_auc = evaluate_metrics(
            model,
            dataloader,
            device,
            head_type=HEAD_TYPE_SIGMOID,
            decision_threshold=args.decision_threshold,
        )
        acc_str, auc_str = format_head_metrics(per_head_acc, per_head_auc)

        print(
            f"Evaluation on {csv_path} ({len(dataset)} samples):\n"
            f"  Avg ACC {avg_acc:.4f} | Avg AUC {avg_auc:.4f}\n"
            f"  Per-head: {acc_str}; {auc_str}"
        )

        report_lines.append(
            f"CSV={csv_path} samples={len(dataset)} | Avg ACC={avg_acc:.4f} Avg AUC={avg_auc:.4f} | "
            f"Per-head ACC: {acc_str} | Per-head AUC: {auc_str}"
        )

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text("\n".join(report_lines))
        print(f"Saved evaluation report to {args.report_path}")


if __name__ == "__main__":
    main()
