import argparse
import re
import subprocess
import sys
from pathlib import Path


PRETRAINED_KEYS = {
    "train_csv",
    "val_csv",
    "image_root",
    "epochs",
    "batch_size",
    "batch_size_grid",
    "lr",
    "lr_grid",
    "min_lr",
    "weight_decay",
    "label_smoothing",
    "num_workers",
    "save_path",
    "search_report",
    "resume_checkpoint",
    "seed",
    "log_dir",
    "grad_accum_steps",
    "grad_clip_norm",
    "amp",
    "freeze_backbone_epochs",
    "backbone_lr",
    "early_stop_patience",
}

SCRATCH_KEYS = {
    "train_csv",
    "val_csv",
    "image_root",
    "epochs",
    "batch_size",
    "batch_size_grid",
    "lr",
    "lr_grid",
    "min_lr",
    "weight_decay",
    "num_workers",
    "save_path",
    "search_report",
    "resume_checkpoint",
    "seed",
    "log_dir",
    "decision_threshold",
    "grad_accum_steps",
    "grad_clip_norm",
    "amp",
    "freeze_backbone_epochs",
    "backbone_lr",
    "early_stop_patience",
}

TEST_PRETRAINED_KEYS = {
    "csv",
    "weights",
    "image_root",
    "batch_size",
    "num_workers",
    "imagenet_init",
    "report_path",
}

TEST_SCRATCH_KEYS = {
    "csv",
    "weights",
    "image_root",
    "batch_size",
    "num_workers",
    "imagenet_init",
    "report_path",
    "decision_threshold",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a training or evaluation script using a YAML config file."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help=(
            "Script path (optional if config name includes 'pretrained', 'scratch', "
            "'test_pretrained', or 'test_scratch')"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved command without running it",
    )
    return parser.parse_args()


def infer_script_from_config(config_path: Path) -> tuple[Path, set[str]]:
    name = config_path.name.lower()
    if "test" in name and "pretrained" in name:
        return Path("test_densenet.py"), TEST_PRETRAINED_KEYS
    if "test" in name and "scratch" in name:
        return Path("test_densenet_no.py"), TEST_SCRATCH_KEYS
    if "pretrained" in name:
        return Path("train_densenet_pretrained.py"), PRETRAINED_KEYS
    if "scratch" in name:
        return Path("train_densenet_scratch.py"), SCRATCH_KEYS
    raise SystemExit(
        "Unable to infer script from config name. "
        "Use --script to specify a train/test script explicitly."
    )


def strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    result = []
    for ch in value:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        result.append(ch)
    return "".join(result).strip()


def parse_scalar(value: str):
    value = strip_inline_comment(value)
    if not value:
        return None

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", value):
        return float(value)

    return value


def parse_config_simple(config_path: Path) -> dict:
    data: dict[str, object] = {}
    current_list_key: str | None = None

    lines = config_path.read_text(encoding="utf-8").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if current_list_key and line.startswith("-"):
            item = line[1:].strip()
            data[current_list_key].append(parse_scalar(item))
            continue

        if ":" not in line:
            raise ValueError(f"Invalid line in config: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = strip_inline_comment(value.strip())

        if not value:
            data[key] = []
            current_list_key = key
            continue

        current_list_key = None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                items = [parse_scalar(item.strip()) for item in inner.split(",")]
                data[key] = items
        else:
            data[key] = parse_scalar(value)

    return data


def load_config(config_path: Path) -> dict:
    try:
        import yaml
    except Exception:
        return parse_config_simple(config_path)

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping of keys to values.")
    return data


def allowed_keys_for_script(script_path: Path) -> set[str]:
    name = script_path.name.lower()
    if "test_densenet_no" in name:
        return TEST_SCRATCH_KEYS
    if "test_densenet" in name:
        return TEST_PRETRAINED_KEYS
    if "train_densenet_pretrained" in name:
        return PRETRAINED_KEYS
    if "train_densenet_scratch" in name:
        return SCRATCH_KEYS
    if "pretrained" in name:
        return PRETRAINED_KEYS
    if "scratch" in name:
        return SCRATCH_KEYS
    raise SystemExit(
        "Unable to infer allowed keys for script. "
        "Use a known train/test script or rename the script accordingly."
    )


def resolve_script_path(script_path: Path, config_path: Path) -> Path:
    if script_path.is_absolute():
        return script_path
    if script_path.exists():
        return script_path
    candidate = config_path.parent / script_path
    if candidate.exists():
        return candidate
    return script_path


def build_command(script_path: Path, config: dict, allowed_keys: set[str]) -> list[str]:
    unknown_keys = [key for key in config if key not in allowed_keys]
    if unknown_keys:
        raise SystemExit(f"Unknown config keys for this script: {', '.join(unknown_keys)}")

    cmd = [sys.executable, str(script_path)]
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            cmd.append(flag)
            cmd.extend(str(item) for item in value)
        else:
            cmd.append(flag)
            cmd.append(str(value))
    return cmd


def main():
    args = parse_args()
    config_path = args.config
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    config = load_config(config_path)
    if args.script is None:
        script_path, allowed_keys = infer_script_from_config(config_path)
    else:
        script_path = args.script
        allowed_keys = allowed_keys_for_script(script_path)

    script_path = resolve_script_path(script_path, config_path)
    cmd = build_command(script_path, config, allowed_keys)

    if args.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
