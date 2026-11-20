"""
Offline-capable DINOv2 Small embedding pipeline.

Model repository: facebook/dinov2-small (Hugging Face)
Architecture: DINOv2 ViT-S/14
Embedding: CLS token, 384 dimensions

One-time download on a machine with internet:
    pip install -U transformers huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/dinov2-small', local_dir='models/dinov2-small')"

Then copy the entire models/dinov2-small directory to this project and use:
    model, device = load_dinov2("models/dinov2-small", local_files_only=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

EMBEDDING_DIM = 384
DEFAULT_HF_REPO = "facebook/dinov2-small"


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dinov2(
    model_path: str | Path = DEFAULT_HF_REPO,
    device: Optional[str] = None,
    local_files_only: bool = True,
):
    """
    Load DINOv2-Small using Hugging Face Transformers.

    For offline use, model_path must point to a previously downloaded
    local snapshot directory. Set local_files_only=True to prevent network calls.
    """
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "Install Transformers first: pip install -U transformers"
        ) from exc

    device_obj = get_device(device)
    model_path = str(model_path)

    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to(device_obj)

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size != EMBEDDING_DIM:
        raise ValueError(
            f"Expected DINOv2-Small hidden size {EMBEDDING_DIM}, got {hidden_size}. "
            "Check that the model directory is facebook/dinov2-small."
        )

    return model, device_obj


def build_transform():
    """ImageNet normalization and 224x224 input preprocessing."""
    return transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


class ImagePathDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Optional[str]],
        transform,
        base_dir: Optional[str | Path] = None,
    ):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        raw_path = self.image_paths[index]
        if raw_path is None or pd.isna(raw_path) or str(raw_path).strip() == "":
            return index, None

        path = Path(str(raw_path))
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / path

        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return index, tensor
        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError, ValueError):
            return index, None


def _collate_images(batch):
    valid = [(idx, image) for idx, image in batch if image is not None]
    if not valid:
        return [], None
    indices, images = zip(*valid)
    return list(indices), torch.stack(images)


def load_images_table(images_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(images_path)
    if "image_path" not in df.columns:
        raise ValueError("The images table must contain an 'image_path' column.")
    return df


def _is_valid_embedding(value) -> bool:
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and len(value) == EMBEDDING_DIM
    )


def extract_embeddings(
    images_df: pd.DataFrame,
    model,
    device: torch.device,
    base_dir: Optional[str | Path] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    existing_embeddings: Optional[Sequence] = None,
    show_progress: bool = True,
) -> list:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    n = len(images_df)
    embeddings = list(existing_embeddings) if existing_embeddings is not None else [None] * n
    if len(embeddings) != n:
        raise ValueError("existing_embeddings length must match images_df length.")

    pending_positions = [
        i for i, emb in enumerate(embeddings) if not _is_valid_embedding(emb)
    ]
    if not pending_positions:
        return embeddings

    pending_paths = images_df.iloc[pending_positions]["image_path"].tolist()
    dataset = ImagePathDataset(pending_paths, build_transform(), base_dir=base_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_images,
    )

    model.eval()
    iterator = tqdm(loader, desc="Extracting DINOv2 embeddings", disable=not show_progress)

    with torch.inference_mode():
        for local_indices, batch in iterator:
            if batch is None:
                continue

            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            outputs = model(pixel_values=batch)

            # Hugging Face DINOv2 returns last_hidden_state:
            # [batch, tokens, 384]. The CLS token is token 0.
            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                features = outputs["last_hidden_state"][:, 0, :]
            elif torch.is_tensor(outputs):
                features = outputs
            else:
                raise RuntimeError("Unsupported model output; cannot extract CLS embeddings.")

            if features.ndim != 2 or features.shape[1] != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Expected [batch, {EMBEDDING_DIM}] features, got {tuple(features.shape)}"
                )

            if normalize:
                features = F.normalize(features, p=2, dim=1)

            features = features.detach().cpu().numpy().astype(np.float32)
            for local_idx, vector in zip(local_indices, features):
                original_idx = pending_positions[local_idx]
                embeddings[original_idx] = vector.tolist()

    return embeddings


def add_embeddings_column(
    images_df: pd.DataFrame,
    embeddings: Sequence,
    column_name: str = "embedding",
) -> pd.DataFrame:
    if len(images_df) != len(embeddings):
        raise ValueError("Number of embeddings must match number of image rows.")
    result = images_df.copy()
    result[column_name] = list(embeddings)
    return result


def save_embeddings_parquet(
    images_df: pd.DataFrame,
    embeddings: Sequence,
    output_path: str | Path,
    column_name: str = "embedding",
):
    result = add_embeddings_column(images_df, embeddings, column_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def run_embedding_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    base_dir: Optional[str | Path] = None,
    model_path: str | Path = DEFAULT_HF_REPO,
    device: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
    save_every_batches: Optional[int] = 20,
    local_files_only: bool = True,
):
    input_path = Path(input_path)
    output_path = Path(output_path)
    images_df = load_images_table(input_path)

    embeddings = [None] * len(images_df)

    # Resume only when row order and identifiers match exactly.
    if output_path.exists():
        previous = pd.read_parquet(output_path)
        identity_cols = [
            c for c in ("image_id", "observation_uuid", "image_path")
            if c in images_df.columns
        ]
        same_rows = (
            len(previous) == len(images_df)
            and "embedding" in previous.columns
            and bool(identity_cols)
            and all(c in previous.columns and previous[c].equals(images_df[c]) for c in identity_cols)
        )
        if same_rows:
            embeddings = previous["embedding"].tolist()
            print(f"Resuming from {sum(_is_valid_embedding(x) for x in embeddings):,} existing embeddings.")

    model, device_obj = load_dinov2(
        model_path=model_path,
        device=device,
        local_files_only=local_files_only,
    )
    print(f"Device: {device_obj}")
    print(f"Images: {len(images_df):,}")

    chunk_size = (
        batch_size * save_every_batches
        if save_every_batches is not None and save_every_batches > 0
        else len(images_df)
    )
    chunk_size = max(chunk_size, batch_size)

    for start in range(0, len(images_df), chunk_size):
        end = min(start + chunk_size, len(images_df))
        chunk_df = images_df.iloc[start:end].reset_index(drop=True)

        chunk_embeddings = extract_embeddings(
            chunk_df,
            model=model,
            device=device_obj,
            base_dir=base_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            existing_embeddings=embeddings[start:end],
        )
        embeddings[start:end] = chunk_embeddings

        save_embeddings_parquet(images_df, embeddings, output_path)

        done = sum(_is_valid_embedding(x) for x in embeddings)
        print(f"Checkpoint: {done:,}/{len(images_df):,}")

    final_df = save_embeddings_parquet(images_df, embeddings, output_path)
    print(f"Saved: {output_path.resolve()}")
    return final_df


def cosine_similarity(vector_a, vector_b) -> float:
    a = np.asarray(vector_a, dtype=np.float32)
    b = np.asarray(vector_b, dtype=np.float32)
    if a.shape != (EMBEDDING_DIM,) or b.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Both vectors must have shape ({EMBEDDING_DIM},).")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def find_similar_images(
    images_df: pd.DataFrame,
    query_image_id,
    top_k: int = 5,
    embedding_column: str = "embedding",
) -> pd.DataFrame:
    if "image_id" not in images_df.columns:
        raise ValueError("images_df must contain image_id.")

    matches = images_df.index[images_df["image_id"] == query_image_id].tolist()
    if not matches:
        raise KeyError(f"image_id={query_image_id!r} was not found.")

    query_vector = images_df.loc[matches[0], embedding_column]
    if not _is_valid_embedding(query_vector):
        raise ValueError("The query image does not have a valid embedding.")

    valid_mask = images_df[embedding_column].apply(_is_valid_embedding)
    candidates = images_df.loc[valid_mask].copy()
    matrix = np.stack(candidates[embedding_column].to_numpy()).astype(np.float32)

    similarities = matrix @ np.asarray(query_vector, dtype=np.float32)
    candidates["similarity"] = similarities
    candidates = candidates[candidates["image_id"] != query_image_id]

    return candidates.sort_values("similarity", ascending=False).head(top_k)
