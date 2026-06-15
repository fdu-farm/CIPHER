from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF

HEAD_TYPE_SOFTMAX = "softmax"
HEAD_TYPE_SIGMOID = "sigmoid"

TRANSFORM_IMAGENET = "imagenet"
TRANSFORM_MEDICAL = "medical_512"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CenterSquareCrop:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return TF.crop(img, top, left, side, side)


class MultiHeadImageDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path | None = None,
        transform=None,
        label_dtype: np.dtype | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.label_dtype = label_dtype if label_dtype is not None else np.float32

        if self.df.shape[1] < 2:
            raise ValueError("CSV must contain at least 2 columns: path + labels")

        self.num_labels = self.df.shape[1] - 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row.iloc[0])
        if self.image_root:
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            row.iloc[1 : 1 + self.num_labels].values.astype(self.label_dtype)
        )
        return image, labels


def label_dtype_for_head_type(head_type: str) -> np.dtype:
    if head_type == HEAD_TYPE_SIGMOID:
        return np.float32
    if head_type == HEAD_TYPE_SOFTMAX:
        return np.int64
    raise ValueError(f"Unsupported head type: {head_type}")


def build_transforms(preset: str, is_train: bool):
    if preset == TRANSFORM_IMAGENET:
        resize_ops = (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
            if is_train
            else [transforms.Resize(256), transforms.CenterCrop(224)]
        )
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return transforms.Compose(
            [
                *resize_ops,
                transforms.ToTensor(),
                normalize,
            ]
        )

    if preset == TRANSFORM_MEDICAL:
        pre_ops = [
            CenterSquareCrop(),
            transforms.Resize((512, 512)),
        ]
        augmentations = []
        if is_train:
            augmentations = []
        return transforms.Compose(
            [
                *pre_ops,
                *augmentations,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.225, 0.225, 0.225],
                ),
            ]
        )

    raise ValueError(f"Unknown transform preset: {preset}")


class MultiHeadDenseNet(nn.Module):
    def __init__(self, num_heads: int, head_type: str, use_imagenet: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_imagenet else None
        backbone = models.densenet121(weights=weights)
        self.features = backbone.features
        self.num_features = backbone.classifier.in_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if head_type == HEAD_TYPE_SOFTMAX:
            out_features = 2
        elif head_type == HEAD_TYPE_SIGMOID:
            out_features = 1
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        self.classifier_heads = nn.ModuleList(
            [nn.Linear(self.num_features, out_features) for _ in range(num_heads)]
        )

    def forward(self, x):
        features = self.features(x)
        features = nn.functional.relu(features, inplace=True)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        return [head(features) for head in self.classifier_heads]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def compute_metrics(
    logits_list: Sequence[torch.Tensor],
    labels: torch.Tensor,
    head_type: str,
    decision_threshold: float = 0.5,
) -> Tuple[float, float, List[float], List[float]]:
    per_head_acc = []
    per_head_auc = []

    for head_logits, label in zip(logits_list, labels.T):
        if head_type == HEAD_TYPE_SIGMOID:
            probs = torch.sigmoid(head_logits.squeeze(-1))
            preds = (probs >= decision_threshold).float()
            per_head_acc.append((preds == label.float()).float().mean().item())
            probs_np = probs.detach().cpu().numpy()
            label_np = label.cpu().numpy()
        elif head_type == HEAD_TYPE_SOFTMAX:
            preds = torch.argmax(head_logits, dim=1)
            per_head_acc.append((preds == label).float().mean().item())
            probs_np = torch.softmax(head_logits, dim=1)[:, 1].detach().cpu().numpy()
            label_np = label.cpu().numpy()
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        if len(np.unique(label_np)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(label_np, probs_np)
        per_head_auc.append(auc)

    return (
        float(np.nanmean(per_head_acc)),
        float(np.nanmean(per_head_auc)),
        per_head_acc,
        per_head_auc,
    )


def compute_loss(
    logits_list: Sequence[torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    head_type: str,
) -> torch.Tensor:
    if head_type == HEAD_TYPE_SIGMOID:
        return sum(
            criterion(logits.squeeze(-1), labels[:, i].float())
            for i, logits in enumerate(logits_list)
        )
    if head_type == HEAD_TYPE_SOFTMAX:
        return sum(criterion(logits, labels[:, i]) for i, logits in enumerate(logits_list))
    raise ValueError(f"Unsupported head type: {head_type}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_type: str,
    decision_threshold: float,
    grad_accum_steps: int = 1,
    grad_clip_norm: float | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    model.train()
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    total_loss = 0.0
    all_logits = [[] for _ in range(len(model.classifier_heads))]
    all_labels = []
    num_batches = len(dataloader)
    final_group = num_batches % grad_accum_steps
    use_amp = scaler is not None and scaler.is_enabled() and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        if final_group and step > num_batches - final_group:
            accum_steps = final_group
        else:
            accum_steps = grad_accum_steps

        with (torch.cuda.amp.autocast() if use_amp else nullcontext()):
            logits_list = model(images)
            loss = compute_loss(logits_list, labels, criterion, head_type)
            loss_to_backward = loss / accum_steps

        if use_amp:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if step % grad_accum_steps == 0 or step == num_batches:
            if grad_clip_norm and grad_clip_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        for i, logits in enumerate(logits_list):
            all_logits[i].append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    epoch_logits = [torch.cat(logits) for logits in all_logits]
    epoch_labels = torch.cat(all_labels)
    acc, auc, per_head_acc, per_head_auc = compute_metrics(
        epoch_logits,
        epoch_labels,
        head_type,
        decision_threshold=decision_threshold,
    )
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, acc, auc, per_head_acc, per_head_auc


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    head_type: str,
    decision_threshold: float,
):
    model.eval()
    total_loss = 0.0
    all_logits = [[] for _ in range(len(model.classifier_heads))]
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits_list = model(images)
            loss = compute_loss(logits_list, labels, criterion, head_type)

            total_loss += loss.item() * images.size(0)
            for i, logits in enumerate(logits_list):
                all_logits[i].append(logits.cpu())
            all_labels.append(labels.cpu())

    epoch_logits = [torch.cat(logits) for logits in all_logits]
    epoch_labels = torch.cat(all_labels)
    acc, auc, per_head_acc, per_head_auc = compute_metrics(
        epoch_logits,
        epoch_labels,
        head_type,
        decision_threshold=decision_threshold,
    )
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, acc, auc, per_head_acc, per_head_auc


def evaluate_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    head_type: str,
    decision_threshold: float,
):
    model.eval()
    all_logits = [[] for _ in range(len(model.classifier_heads))]
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits_list = model(images)
            for i, logits in enumerate(logits_list):
                all_logits[i].append(logits.cpu())
            all_labels.append(labels.cpu())

    epoch_logits = [torch.cat(logits) for logits in all_logits]
    epoch_labels = torch.cat(all_labels)
    return compute_metrics(
        epoch_logits,
        epoch_labels,
        head_type,
        decision_threshold=decision_threshold,
    )


def format_head_metrics(per_head_acc: Sequence[float], per_head_auc: Sequence[float]):
    acc_str = ", ".join(f"H{i+1} ACC {acc:.4f}" for i, acc in enumerate(per_head_acc))
    auc_str = ", ".join(
        f"H{i+1} AUC {auc:.4f}" if not np.isnan(auc) else f"H{i+1} AUC nan"
        for i, auc in enumerate(per_head_auc)
    )
    return acc_str, auc_str


def log_epoch_metrics(
    writer,
    epoch: int,
    phase: str,
    loss: float,
    acc: float,
    auc: float,
    per_head_acc: Sequence[float],
    per_head_auc: Sequence[float],
) -> None:
    writer.add_scalar(f"{phase}/loss", loss, epoch)
    writer.add_scalar(f"{phase}/acc", acc, epoch)
    writer.add_scalar(f"{phase}/auc", auc, epoch)
    for idx, head_acc in enumerate(per_head_acc, start=1):
        writer.add_scalar(f"{phase}/head_{idx}_acc", head_acc, epoch)
    for idx, head_auc in enumerate(per_head_auc, start=1):
        writer.add_scalar(f"{phase}/head_{idx}_auc", head_auc, epoch)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    load_result = model.load_state_dict(state_dict, strict=False)
    return checkpoint, load_result
