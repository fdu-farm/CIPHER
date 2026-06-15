"""CIPHER Path 3/4 single-image counterfactual editor.

Edit sensitive-attribute tokens while keeping disease words fixed to realize
Path 3. Edit disease words while keeping subgroup metadata fixed to realize
Path 4. The same inversion trajectory preserves patient-specific structure.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional

import yaml

import numpy as np
import torch
import torch.nn.functional as F
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
DEFAULT_OUTPUT = "generated_image_nulltext.png"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_NULL_TEXT_STEPS = 5
DEFAULT_NULL_TEXT_LR = 1e-2
DEFAULT_TORCH_DTYPE = "auto"
DEFAULT_SCHEDULER = "ddim"
DEFAULT_RESOLUTION = 512
SCHEDULER_CHOICES = ("ddim", "dpmpp", "euler_a", "unipc")
DEFAULT_SEG_MODEL = "attunet"
DEFAULT_SEG_INPUT_SIZE = 224
DEFAULT_SEG_THRESHOLD = 0.5
DEFAULT_EDIT_TARGET = "lesion"
DEFAULT_SEG_CLASS_INDEX = 1
SEG_MODEL_CHOICES = ("attunet", "unet", "unetpp", "multiresunet", "resunet")
DEFAULT_SEG_REPO = Path(__file__).resolve().parent.parent / "Awesome-U-Net-main"


SEG_MODEL_REGISTRY = {
    "attunet": ("models.attunet", "AttU_Net", {"img_ch": 3, "output_ch": 2}),
    "unet": ("models.unet", "UNet", {"in_channels": 3, "out_channels": 2, "with_bn": False}),
    "unetpp": ("models.unetpp", "NestedUNet", {"num_classes": 2, "input_channels": 3, "deep_supervision": False}),
    "multiresunet": ("models.multiresunet", "MultiResUnet", {"channels": 3, "filters": 32, "nclasses": 2}),
    "resunet": ("models._resunet.res_unet", "ResUnet", {"in_ch": 3, "out_ch": 2}),
}


def load_config(path: str) -> dict:
    with open(path, "r") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")
    return config


def resolve_seg_repo(repo_value: Optional[str]) -> Path:
    if repo_value:
        return Path(repo_value)
    return DEFAULT_SEG_REPO


def load_segmentation_model(
    name: str,
    checkpoint: str,
    device: torch.device,
    repo_path: Optional[Path] = None,
) -> torch.nn.Module:
    if name not in SEG_MODEL_REGISTRY:
        raise ValueError(f"Unsupported segmentation model '{name}'. Choices: {', '.join(SEG_MODEL_REGISTRY)}")
    repo = resolve_seg_repo(repo_path)
    if not repo.exists():
        raise FileNotFoundError(f"Segmentation repo not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    module_path, class_name, kwargs = SEG_MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    model = model_cls(**kwargs)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def segment_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int,
    threshold: float,
    class_index: int,
) -> torch.Tensor:
    resized = image.resize((input_size, input_size), resample=Image.BILINEAR)
    array = np.array(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=1)
        channel = min(class_index, probs.shape[1] - 1)
        probs = probs[:, channel : channel + 1]
    mask = (probs > threshold).float()
    mask = F.interpolate(mask, size=(image.height, image.width), mode="nearest")
    mask = torch.clamp(mask, 0.0, 1.0)
    return mask.squeeze(0).squeeze(0).detach()


def prepare_latent_mask(mask: Optional[torch.Tensor], latents: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = mask.to(device=latents.device, dtype=latents.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
    mask = torch.clamp(mask, 0.0, 1.0)
    return mask


def blend_latents(latents: torch.Tensor, base_latents: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return latents
    return base_latents * (1 - mask) + latents * mask


def build_background_schedule(
    latents_list: List[torch.Tensor], timesteps: torch.Tensor, skip_steps: int
) -> List[torch.Tensor]:
    schedule = list(reversed(latents_list))
    if skip_steps > 0:
        schedule = schedule[skip_steps:]
    target_length = len(timesteps)
    if len(schedule) < target_length:
        raise ValueError(
            f"Background schedule shorter than timesteps: {len(schedule)} vs {target_length}"
        )
    return [latent.detach() for latent in schedule[:target_length]]


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


def build_scheduler(name: str, pipeline: DiffusionPipeline) -> DDIMScheduler:
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


def preprocess_image(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor * 2 - 1


def image_to_latents(
    vae: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
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


def ddim_step(
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    sample: torch.Tensor,
    scheduler: DDIMScheduler,
) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(sample.device)
    t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
    t_next = (
        int(next_timestep.item()) if isinstance(next_timestep, torch.Tensor) else int(next_timestep)
    )
    alpha_prod_t = alphas_cumprod[t].to(sample.dtype)
    if t_next < 0:
        alpha_prod_t_next = scheduler.final_alpha_cumprod.to(sample.device).to(sample.dtype)
    else:
        alpha_prod_t_next = alphas_cumprod[t_next].to(sample.dtype)
    beta_prod_t = 1 - alpha_prod_t

    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
    elif prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")

    if getattr(scheduler.config, "thresholding", False):
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif getattr(scheduler.config, "clip_sample", False):
        clip_range = getattr(scheduler.config, "clip_sample_range", 1.0)
        pred_original_sample = pred_original_sample.clamp(-clip_range, clip_range)

    pred_sample_direction = (1 - alpha_prod_t_next).sqrt() * pred_epsilon
    next_sample = alpha_prod_t_next.sqrt() * pred_original_sample + pred_sample_direction
    return next_sample


def ddim_inversion(
    unet: torch.nn.Module,
    scheduler: DDIMScheduler,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> List[torch.Tensor]:
    timesteps = scheduler.timesteps
    timesteps_inv = list(reversed(timesteps))
    latents_list = [latents]

    with torch.no_grad():
        for idx, t in enumerate(timesteps_inv[:-1]):
            next_t = timesteps_inv[idx + 1]
            latent_model_input = scale_model_input(scheduler, latents, t)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            latents = ddim_step(noise_pred, t, next_t, latents, scheduler)
            latents_list.append(latents)

    return latents_list


def null_text_optimization(
    unet: torch.nn.Module,
    scheduler: DDIMScheduler,
    latents_list: List[torch.Tensor],
    uncond_embed: torch.Tensor,
    cond_embed: torch.Tensor,
    guidance_scale: float,
    num_opt_steps: int,
    lr: float,
) -> List[torch.Tensor]:
    timesteps = scheduler.timesteps
    uncond_embeddings = []
    latents = latents_list[-1]

    for i, t in enumerate(timesteps[:-1]):
        target_latent = latents_list[-i - 2]
        next_t = timesteps[i + 1]
        uncond_step = uncond_embed.detach().clone()
        uncond_step.requires_grad_(True)
        optimizer = torch.optim.Adam([uncond_step], lr=lr)

        for _ in range(num_opt_steps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scale_model_input(scheduler, latent_model_input, t)
            text_embeddings = torch.cat([uncond_step, cond_embed])
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            prev_latents = ddim_step(guided_noise, t, next_t, latents, scheduler)
            loss = torch.mean((prev_latents - target_latent) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([uncond_step], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            uncond_step.data = torch.nan_to_num(
                uncond_step.data, nan=0.0, posinf=0.0, neginf=0.0
            )

        uncond_embeddings.append(uncond_step.detach())

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scale_model_input(scheduler, latent_model_input, t)
            text_embeddings = torch.cat([uncond_step.detach(), cond_embed])
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latents = ddim_step(guided_noise, t, next_t, latents, scheduler)
    uncond_embeddings.append(uncond_embed.detach())
    return uncond_embeddings


def generate_with_null_text(
    unet: torch.nn.Module,
    scheduler,
    latents: torch.Tensor,
    uncond_embeddings: List[torch.Tensor],
    cond_embeddings: torch.Tensor,
    guidance_scale: float,
    use_ddim_step: bool,
    mask: Optional[torch.Tensor] = None,
    base_latents_per_step: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    timesteps = scheduler.timesteps
    if base_latents_per_step is not None and len(base_latents_per_step) < len(timesteps):
        raise ValueError(
            f"Background schedule shorter than timesteps: {len(base_latents_per_step)} vs {len(timesteps)}"
        )

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scale_model_input(scheduler, latent_model_input, t)
            text_embeddings = torch.cat([uncond_embeddings[i], cond_embeddings])
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            guided_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            if use_ddim_step:
                next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                latents = ddim_step(guided_noise, t, next_t, latents, scheduler)
            else:
                latents = scheduler.step(guided_noise, t, latents).prev_sample
            if mask is not None and base_latents_per_step is not None:
                background = base_latents_per_step[i]
                background = background.to(device=latents.device, dtype=latents.dtype)
                latents = blend_latents(latents, background, mask)
    return latents

def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Path to config YAML.")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config) if config_args.config else {}

    parser = argparse.ArgumentParser(
        description="Image + prompt generation using Null-Text inversion.",
        parents=[config_parser],
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    parser.add_argument("--input-image", default=None, help="Path to the input image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Target prompt.")
    parser.add_argument(
        "--inversion-prompt",
        default=None,
        help="Prompt used for inversion; defaults to --prompt.",
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
        "--null-text-steps",
        type=int,
        default=DEFAULT_NULL_TEXT_STEPS,
        help="Optimization steps per timestep.",
    )
    parser.add_argument(
        "--null-text-lr",
        type=float,
        default=DEFAULT_NULL_TEXT_LR,
        help="Learning rate for Null-Text optimization.",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=0,
        help="Skip early steps to preserve more of the input image.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default=DEFAULT_SCHEDULER,
        help="Scheduler for the editing pass; DDIM is recommended for Null-Text.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Resize input image to a square resolution if width/height are not set.",
    )
    parser.add_argument("--height", type=int, default=None, help="Override input height.")
    parser.add_argument("--width", type=int, default=None, help="Override input width.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path.")
    parser.add_argument(
        "--seg-checkpoint",
        default=None,
        help="Path to Awesome-U-Net weights (.pt/.pth) for lesion segmentation.",
    )
    parser.add_argument(
        "--seg-model",
        choices=SEG_MODEL_CHOICES,
        default=DEFAULT_SEG_MODEL,
        help="Segmentation backbone from Awesome-U-Net-main.",
    )
    parser.add_argument(
        "--seg-input-size",
        type=int,
        default=DEFAULT_SEG_INPUT_SIZE,
        help="Resize the image to this square size before segmentation.",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=DEFAULT_SEG_THRESHOLD,
        help="Probability threshold for the lesion mask.",
    )
    parser.add_argument(
        "--seg-class-index",
        type=int,
        default=DEFAULT_SEG_CLASS_INDEX,
        help="Which class channel to treat as lesion when the model outputs >1 channels.",
    )
    parser.add_argument(
        "--seg-repo",
        default=None,
        help="Path to Awesome-U-Net-main if it's not alongside this repo.",
    )
    parser.add_argument(
        "--edit-target",
        choices=("lesion", "normal", "all"),
        default=DEFAULT_EDIT_TARGET,
        help="Which region to edit: lesion only, background/normal skin, or the whole image.",
    )
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
    if config:
        known_args = {action.dest for action in parser._actions}
        filtered = {key: value for key, value in config.items() if key in known_args}
        parser.set_defaults(**filtered)
        unknown = sorted(set(config) - known_args)
        if unknown:
            print(f"Warning: ignoring unknown config keys: {', '.join(unknown)}")
    args = parser.parse_args()
    if not args.input_image:
        parser.error("--input-image is required (set it in the config or CLI).")
    return args


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    pipeline = load_pipeline(args.model_path, device, dtype)
    ddim_scheduler = build_scheduler("ddim", pipeline)
    ddim_scheduler.set_timesteps(args.num_inference_steps, device=device)

    edit_scheduler = build_scheduler(args.scheduler, pipeline)
    edit_scheduler.set_timesteps(args.num_inference_steps, device=device)
    use_ddim_step = isinstance(edit_scheduler, DDIMScheduler)
    if not use_ddim_step:
        print(
            f"Warning: using {args.scheduler} for editing while inversion/optimization uses DDIM; "
            "results may differ from standard Null-Text."
        )

    seg_model = None
    if args.seg_checkpoint:
        seg_model = load_segmentation_model(
            args.seg_model, args.seg_checkpoint, device, repo_path=args.seg_repo
        )
    elif args.edit_target != "all":
        print("Warning: --seg-checkpoint not provided; editing the full image.")

    image = load_image(args.input_image, args.width, args.height, args.resolution)
    raw_mask = None
    if seg_model is not None:
        raw_mask = segment_image(
            image,
            seg_model,
            device,
            args.seg_input_size,
            args.seg_threshold,
            args.seg_class_index,
        )
        if args.edit_target == "normal":
            raw_mask = 1.0 - raw_mask
        elif args.edit_target == "all":
            raw_mask = torch.ones_like(raw_mask)
        coverage = raw_mask.mean().item()
        if coverage < 1e-5:
            print("Warning: segmentation mask is empty; editing the full image instead.")
            raw_mask = torch.ones_like(raw_mask)
    image_tensor = preprocess_image(image, device, dtype)
    with torch.no_grad():
        image_latents = image_to_latents(pipeline.vae, image_tensor)
    latent_mask = prepare_latent_mask(raw_mask, image_latents) if raw_mask is not None else None

    inversion_prompt = args.inversion_prompt or args.prompt
    cond_embed_inv = encode_prompt(
        pipeline.tokenizer, pipeline.text_encoder, inversion_prompt, device, dtype
    )
    cond_embed_edit = encode_prompt(
        pipeline.tokenizer, pipeline.text_encoder, args.prompt, device, dtype
    )
    uncond_embed = encode_prompt(
        pipeline.tokenizer, pipeline.text_encoder, "", device, dtype
    )

    latents_list = ddim_inversion(pipeline.unet, ddim_scheduler, image_latents, cond_embed_inv)
    uncond_embeddings = null_text_optimization(
        pipeline.unet,
        ddim_scheduler,
        latents_list,
        uncond_embed,
        cond_embed_inv,
        args.guidance_scale,
        args.null_text_steps,
        args.null_text_lr,
    )

    skip_steps = max(0, min(args.skip_steps, len(edit_scheduler.timesteps) - 1))
    timesteps = edit_scheduler.timesteps[skip_steps:]
    edit_scheduler.timesteps = timesteps
    uncond_embeddings = uncond_embeddings[skip_steps:]
    background_schedule = None
    if latent_mask is not None:
        background_schedule = build_background_schedule(latents_list, timesteps, skip_steps)
    start_latents = (
        background_schedule[0] if background_schedule is not None else latents_list[-(skip_steps + 1)]
    )

    final_latents = generate_with_null_text(
        pipeline.unet,
        edit_scheduler,
        start_latents,
        uncond_embeddings,
        cond_embed_edit,
        args.guidance_scale,
        use_ddim_step,
        mask=latent_mask,
        base_latents_per_step=background_schedule,
    )
    output_image = latents_to_image(pipeline.vae, final_latents)
    output_image.save(args.output)
    print(f"Image saved at {args.output}")


if __name__ == "__main__":
    main()













