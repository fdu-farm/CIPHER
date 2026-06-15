import argparse
import csv
from pathlib import Path
from typing import Optional

# Unified batch runner for assembling mixed-path CIPHER augmentation sets.
# Rows without an input image behave like global Path 1/2 synthesis.
# Rows with an input image behave like image-conditioned augmentation that can
# be mixed with the instance-anchored Path 3/4 editing outputs.

import yaml
import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
)
from PIL import Image

DEFAULT_MODEL_PATH = (
    "./checkpoints/cipher"
)
DEFAULT_PROMPT = (
    "45 years old female with light skin tone. Dermoscopic image of a benign "
    "melanocytic nevus with light-to-medium brown pigmentation, a fairly symmetric "
    "pattern, and smooth lesion borders."
)
DEFAULT_OUTPUT_DIR = "outputs/unified_batch"
DEFAULT_OUTPUT_NAME = "image_{index:05d}.png"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_STRENGTH = 0.5
DEFAULT_TORCH_DTYPE = "auto"
DEFAULT_SCHEDULER = "dpmpp"
DEFAULT_RESOLUTION = 512
SCHEDULER_CHOICES = ("ddim", "dpmpp", "euler_a", "unipc")


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


def resolve_output_name(output_name: Optional[str], image_path: Optional[Path], index: int) -> str:
    if not is_blank(output_name):
        return str(output_name).format(index=index)
    if image_path is not None:
        return f"{image_path.stem}_{index:05d}.png"
    return DEFAULT_OUTPUT_NAME.format(index=index)


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


def load_pipeline(model_path: str, device: torch.device, dtype: torch.dtype) -> DiffusionPipeline:
    pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    required = ("vae", "unet", "text_encoder", "tokenizer", "scheduler")
    missing = [name for name in required if not hasattr(pipeline, name)]
    if missing:
        raise ValueError(f"Pipeline is missing required components: {', '.join(missing)}")
    pipeline.vae.eval()
    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    return pipeline


def build_scheduler(name: str, pipeline: DiffusionPipeline):
    if name == "ddim":
        scheduler_cls = DDIMScheduler
    elif name == "dpmpp":
        scheduler_cls = DPMSolverMultistepScheduler
    elif name == "euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif name == "unipc":
        scheduler_cls = UniPCMultistepScheduler
    else:
        scheduler_cls = DDIMScheduler
    return scheduler_cls.from_config(pipeline.scheduler.config)


def enable_optimizations(pipeline: DiffusionPipeline, attention_slicing: bool, xformers: bool) -> None:
    if attention_slicing:
        pipeline.enable_attention_slicing()
    if xformers:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Warning: xformers not available ({exc}).")


def load_image(
    path: str,
    width: Optional[int],
    height: Optional[int],
    resolution: Optional[int],
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if width is None and height is None:
        resolution = DEFAULT_RESOLUTION if resolution is None else resolution
        width = resolution
        height = resolution
    else:
        width = image.width if width is None else width
        height = image.height if height is None else height
    width = max(8, width - width % 8)
    height = max(8, height - height % 8)
    if image.size != (width, height):
        orig_w, orig_h = image.size
        scale = min(width / orig_w, height / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resized = image.resize((new_w, new_h), resample=Image.LANCZOS)
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        left = (width - new_w) // 2
        top = (height - new_h) // 2
        canvas.paste(resized, (left, top))
        image = canvas
    return image


def resolve_dimensions(
    width: Optional[int],
    height: Optional[int],
    resolution: Optional[int],
) -> tuple[int, int]:
    if width is None and height is None:
        resolution = DEFAULT_RESOLUTION if resolution is None else resolution
        width = resolution
        height = resolution
    else:
        if width is None:
            width = height
        if height is None:
            height = width
    width = max(8, width - width % 8)
    height = max(8, height - height % 8)
    return width, height


def preprocess_image(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor * 2 - 1


def image_to_latents(vae: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    posterior = vae.encode(image_tensor).latent_dist
    latents = posterior.sample()
    return latents * scaling_factor


def latents_to_image(vae: torch.nn.Module, latents: torch.Tensor) -> Image.Image:
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = latents / scaling_factor
    latents = torch.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(dtype=vae_dtype)
    with torch.no_grad():
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        embeddings = text_encoder(input_ids)[0]
    return embeddings.to(dtype)


def scale_model_input(scheduler, latent_model_input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    if hasattr(scheduler, "scale_model_input"):
        return scheduler.scale_model_input(latent_model_input, timestep)
    return latent_model_input


def prepare_timesteps(
    scheduler,
    num_inference_steps: int,
    strength: float,
    device: torch.device,
) -> torch.Tensor:
    scheduler.set_timesteps(num_inference_steps, device=device)
    strength = max(0.0, min(1.0, strength))
    if strength <= 0:
        return scheduler.timesteps[:0]
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return scheduler.timesteps[t_start:]


def add_noise(
    scheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if timesteps.numel() == 0:
        return latents
    noise = torch.randn(
        latents.shape, generator=generator, device=latents.device, dtype=latents.dtype
    )
    return scheduler.add_noise(latents, noise, timesteps[:1])


def build_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(seed)


def prepare_noise_latents(
    unet: torch.nn.Module,
    scheduler,
    width: int,
    height: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents_shape = (1, unet.in_channels, height // 8, width // 8)
    latents = torch.randn(latents_shape, generator=generator, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


def rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_rescale: float,
) -> torch.Tensor:
    if guidance_rescale <= 0:
        return noise_cfg
    dims = list(range(1, noise_cfg.ndim))
    std_text = noise_pred_text.std(dim=dims, keepdim=True)
    std_cfg = noise_cfg.std(dim=dims, keepdim=True)
    noise_rescaled = noise_cfg * (std_text / (std_cfg + 1e-6))
    return guidance_rescale * noise_rescaled + (1 - guidance_rescale) * noise_cfg


def denoise_latents(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    guidance_rescale: Optional[float],
) -> torch.Tensor:
    with torch.no_grad():
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scale_model_input(scheduler, latent_model_input, t)
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            if guidance_rescale is not None and guidance_rescale > 0:
                guided_noise = rescale_noise_cfg(
                    guided_noise, noise_pred_cond, guidance_rescale
                )
            latents = scheduler.step(guided_noise, t, latents).prev_sample
    return latents


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to config YAML.")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Unified text-to-image and image-to-image batch generation from CSV.",
        parents=[config_parser],
    )
    parser.add_argument("--csv", required=True, help="CSV file with prompt/image columns.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output name template (supports {index}); leave blank to use image stems.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Fallback prompt.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt to suppress unwanted features.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help="How much to deviate from the input image (0-1).",
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
        help="Denoising steps.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default=DEFAULT_SCHEDULER,
        help="Scheduler for denoising.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Fallback resolution when no image is provided.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override height.")
    parser.add_argument("--width", type=int, default=None, help="Override width.")
    parser.add_argument(
        "--guidance-rescale",
        type=float,
        default=None,
        help="Optional CFG rescale if supported.",
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
    scheduler = build_scheduler(args.scheduler, pipeline)
    enable_optimizations(pipeline, args.attention_slicing, args.xformers)

    csv_path = Path(args.csv)
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError("CSV contains no rows.")
    if "prompt" not in rows[0]:
        raise ValueError("CSV must contain a 'prompt' column.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = csv_path.parent

    for idx, row in enumerate(rows, start=1):
        image_value = row.get("image") or row.get("input_image")
        has_image = not is_blank(image_value)
        image_path = resolve_path(str(image_value), base_dir) if has_image else None

        prompt = row.get("prompt")
        if is_blank(prompt):
            prompt = args.prompt

        negative_prompt = row.get("negative_prompt", args.negative_prompt)
        strength = parse_optional_float(row.get("strength"), args.strength)
        guidance_scale = parse_optional_float(row.get("guidance_scale"), args.guidance_scale)
        num_steps = parse_optional_int(row.get("num_inference_steps"), args.num_inference_steps)
        height = parse_optional_int(row.get("height"), args.height)
        width = parse_optional_int(row.get("width"), args.width)
        resolution = parse_optional_int(row.get("resolution"), args.resolution)
        guidance_rescale = parse_optional_float(row.get("guidance_rescale"), args.guidance_rescale)
        seed = parse_optional_int(row.get("seed"), args.seed)

        generator = build_generator(seed, device)

        if has_image:
            timesteps = prepare_timesteps(scheduler, num_steps, strength, device)
            image = load_image(str(image_path), width, height, resolution)
            image_tensor = preprocess_image(image, device, dtype)
            with torch.no_grad():
                latents = image_to_latents(pipeline.vae, image_tensor)
            latents = add_noise(scheduler, latents, timesteps, generator)
        else:
            scheduler.set_timesteps(num_steps, device=device)
            timesteps = scheduler.timesteps
            resolved_width, resolved_height = resolve_dimensions(width, height, resolution)
            latents = prepare_noise_latents(
                pipeline.unet,
                scheduler,
                resolved_width,
                resolved_height,
                generator,
                device,
                dtype,
            )

        cond_embeddings = encode_prompt(
            pipeline.tokenizer, pipeline.text_encoder, prompt, device, dtype
        )
        uncond_embeddings = encode_prompt(
            pipeline.tokenizer, pipeline.text_encoder, negative_prompt, device, dtype
        )

        latents = denoise_latents(
            pipeline.unet,
            scheduler,
            latents,
            timesteps,
            cond_embeddings,
            uncond_embeddings,
            guidance_scale,
            guidance_rescale,
        )
        output_image = latents_to_image(pipeline.vae, latents)

        output_name = resolve_output_name(args.output_name, image_path, idx)
        output_path = resolve_output_path(row.get("output"), output_dir, output_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path)
        mode = "img2img" if has_image else "t2i"
        print(f"[{idx}/{len(rows)}] {mode} saved: {output_path}")


if __name__ == "__main__":
    main()
