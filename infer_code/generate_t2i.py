"""CIPHER Path 1/2 single-image generator.

Use a disease-only prompt to approximate Path 1.
Keep disease wording fixed and swap subgroup metadata tokens to realize Path 2.
"""

import argparse
import inspect
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
DEFAULT_OUTPUT = "generated_image_80_full_pl.png"
DEFAULT_GUIDANCE_SCALE = 12.0
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_SCHEDULER = "dpmpp"
DEFAULT_TORCH_DTYPE = "auto"
SCHEDULER_CHOICES = ("default", "dpmpp", "euler_a", "unipc")


def load_config(path: str) -> dict:
    with open(path, "r") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return config


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
    call_params = inspect.signature(pipeline.__call__).parameters
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
        description="Generate an image from a text prompt.",
        parents=[config_parser],
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text.")
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
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
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

    generator = build_generator(args.seed, device)
    generation_kwargs = prepare_generation_kwargs(
        pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        guidance_rescale=args.guidance_rescale,
        generator=generator,
    )
    image = generate_image(pipeline, generation_kwargs)
    image.save(args.output)
    print(f"Image saved at {args.output}")


if __name__ == "__main__":
    main()
