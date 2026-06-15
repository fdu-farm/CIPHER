"""CIPHER Path 1/2 batch generator.

Each CSV row can represent either disease-only synthesis (Path 1) or
disease-fixed subgroup reshaping (Path 2), depending on the prompt tokens used.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import yaml
import torch
from diffusers import DiffusionPipeline
from PIL import Image

DEFAULT_MODEL_PATH = (
    "./checkpoints/cipher"
)
DEFAULT_PROMPT = (
    "45 years old female with light skin tone. Dermoscopic image of a benign "
    "melanocytic nevus with light-to-medium brown pigmentation, a fairly symmetric "
    "pattern, and smooth lesion borders."
)
DEFAULT_OUTPUT_DIR = "outputs/t2i_batch"
DEFAULT_OUTPUT_NAME = "image_{index:05d}.png"
DEFAULT_GUIDANCE_SCALE = 12.0
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_SCHEDULER = "dpmpp"
DEFAULT_TORCH_DTYPE = "auto"
SCHEDULER_CHOICES = ("default", "dpmpp", "euler_a", "unipc")
DEFAULT_DISABLE_SAFETY_CHECKER = False


def load_config(path: str) -> dict:
    with open(path, "r") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return config


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({(key or "").strip().lower(): value for key, value in row.items()})
    return rows


def is_blank(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "null"}


def parse_optional_int(value: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if is_blank(value):
        return fallback
    return int(value)


def parse_optional_float(value: Optional[str], fallback: Optional[float]) -> Optional[float]:
    if is_blank(value):
        return fallback
    return float(value)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def resolve_output_path(output_value: Optional[str], output_dir: Path, name: str) -> Path:
    if not is_blank(output_value):
        output_path = Path(str(output_value))
        return output_path if output_path.is_absolute() else output_dir / output_path
    return output_dir / name


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return mapping[dtype_name]


def load_pipeline(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> DiffusionPipeline:
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    return pipeline.to(device)


def configure_scheduler(pipeline: DiffusionPipeline, scheduler_name: str) -> None:
    if scheduler_name == "default":
        return
    if not hasattr(pipeline, "scheduler"):
        print("Warning: pipeline has no scheduler; keeping default.")
        return

    if scheduler_name == "dpmpp":
        from diffusers import DPMSolverMultistepScheduler

        scheduler_cls = DPMSolverMultistepScheduler
    elif scheduler_name == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler

        scheduler_cls = EulerAncestralDiscreteScheduler
    elif scheduler_name == "unipc":
        from diffusers import UniPCMultistepScheduler

        scheduler_cls = UniPCMultistepScheduler
    else:
        return

    pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)


def disable_safety_checker(pipeline: DiffusionPipeline) -> None:
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
        if hasattr(pipeline, "requires_safety_checker"):
            pipeline.requires_safety_checker = False
        print("Safety checker disabled.")
    else:
        print("Safety checker not present; nothing to disable.")


def enable_optimizations(
    pipeline: DiffusionPipeline,
    attention_slicing: bool,
    xformers: bool,
) -> None:
    if attention_slicing:
        pipeline.enable_attention_slicing()
    if xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - depends on optional package
            print(f"Warning: xformers not available ({exc}).")


def build_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def prepare_generation_kwargs(
    pipeline: DiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    height: Optional[int],
    width: Optional[int],
    guidance_rescale: Optional[float],
    generator: Optional[torch.Generator],
) -> dict:
    call_params = pipeline.__call__.__code__.co_varnames
    kwargs = {"prompt": prompt}

    def maybe_add(name: str, value: Optional[object]) -> None:
        if name in call_params and value is not None:
            kwargs[name] = value

    maybe_add("negative_prompt", negative_prompt or None)
    maybe_add("guidance_scale", guidance_scale)
    maybe_add("num_inference_steps", num_inference_steps)
    maybe_add("height", height)
    maybe_add("width", width)
    maybe_add("guidance_rescale", guidance_rescale)
    maybe_add("generator", generator)
    return kwargs


def generate_image(pipeline: DiffusionPipeline, kwargs: dict) -> Image.Image:
    result = pipeline(**kwargs)
    return result.images[0]


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to config YAML.")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Batch text-to-image generation from CSV prompts.",
        parents=[config_parser],
    )
    parser.add_argument("--csv", required=True, help="CSV file with at least a prompt column.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output name template (supports {index}).",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fallback prompt.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt to suppress unwanted features.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Classifier-free guidance.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Denoising steps; higher usually improves quality.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default=DEFAULT_SCHEDULER,
        help="Sampling scheduler; DPM++ is a strong default.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override output height.")
    parser.add_argument("--width", type=int, default=None, help="Override output width.")
    parser.add_argument(
        "--guidance-rescale",
        type=float,
        default=None,
        help="Optional CFG rescale if supported by the pipeline.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "fp16", "bf16", "fp32"),
        default=DEFAULT_TORCH_DTYPE,
        help="Weights precision; auto uses fp16 on CUDA, fp32 otherwise.",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
        help="Reduce memory use at a small speed cost.",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Use xformers attention if installed.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        default=DEFAULT_DISABLE_SAFETY_CHECKER,
        help="Disable NSFW safety checker if present in the pipeline.",
    )
    if config:
        known_args = {action.dest for action in parser._actions}
        filtered = {key: value for key, value in config.items() if key in known_args}
        parser.set_defaults(**filtered)
        unknown = sorted(set(config) - known_args)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    print(f"Using device: {device}, dtype: {dtype}")

    pipeline = load_pipeline(args.model_path, device, dtype)
    configure_scheduler(pipeline, args.scheduler)
    enable_optimizations(pipeline, args.attention_slicing, args.xformers)
    if args.disable_safety_checker:
        disable_safety_checker(pipeline)

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError("CSV contains no rows.")
    if "prompt" not in rows[0]:
        raise ValueError("CSV must contain a 'prompt' column.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        if is_blank(prompt):
            prompt = args.prompt
        negative_prompt = row.get("negative_prompt", args.negative_prompt)
        guidance_scale = parse_optional_float(row.get("guidance_scale"), args.guidance_scale)
        num_steps = parse_optional_int(row.get("num_inference_steps"), args.num_inference_steps)
        height = parse_optional_int(row.get("height"), args.height)
        width = parse_optional_int(row.get("width"), args.width)
        guidance_rescale = parse_optional_float(row.get("guidance_rescale"), args.guidance_rescale)
        seed = parse_optional_int(row.get("seed"), args.seed)

        generator = build_generator(seed, device)
        output_name = args.output_name.format(index=idx)
        output_path = resolve_output_path(row.get("output"), output_dir, output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generation_kwargs = prepare_generation_kwargs(
            pipeline,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance_rescale=guidance_rescale,
            generator=generator,
        )
        image = generate_image(pipeline, generation_kwargs)
        image.save(output_path)
        print(f"[{idx}/{len(rows)}] saved: {output_path}")


if __name__ == "__main__":
    main()
