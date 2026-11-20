"""
Feature engineering for wildlife image embeddings and observation metadata.

Expected inputs:
- images.parquet: image_id, observation_uuid, image_path, ...
- observations.parquet: uuid, species_id, observed_on, latitude, longitude, ...
- species.parquet: species_id and taxonomic metadata
- embeddings parquet: image_id plus one vector column (usually embedding/embeddings)

The module creates image-level features:
- local_density: mean cosine similarity to the k nearest *other* images
- visual_isolation: cosine distance to the single nearest other image
- species_visual_spread: mean cosine distance of images in a species to that
  species' normalized embedding centroid
- geographic diversity features, calculated per species from unique observations
- temporal diversity features, calculated per species from unique observations

Geographic/temporal species features are descriptive group summaries. Do not use
them as predictive inputs in a train/test experiment unless they are recomputed
inside each training fold to avoid information leakage.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


EMBEDDING_CANDIDATES = (
    "embedding", "embeddings", "image_embedding", "dinov2_embedding",
    "vector", "features", "feature_vector",
)


def _parse_vector(value) -> np.ndarray:
    """Convert common Parquet vector representations into a 1-D float array."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    elif hasattr(value, "as_py"):
        arr = np.asarray(value.as_py())
    elif isinstance(value, str):
        try:
            arr = np.asarray(json.loads(value))
        except Exception:
            arr = np.asarray(ast.literal_eval(value))
    else:
        arr = np.asarray(value)

    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("Embedding contains an empty vector or NaN/Inf values.")
    return arr


def _get_embedding_matrix(
    embedding_df: pd.DataFrame,
    embedding_col: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """Find the embedding column or a set of embedding_0 ... embedding_n columns."""
    if embedding_col is not None:
        if embedding_col not in embedding_df.columns:
            raise KeyError(
                f"embedding_col={embedding_col!r} not found. "
                f"Available columns: {embedding_df.columns.tolist()}"
            )
        vectors = [_parse_vector(v) for v in embedding_df[embedding_col]]
        matrix = np.vstack(vectors)
        return matrix, embedding_col

    lower_to_original = {str(c).lower(): c for c in embedding_df.columns}
    for candidate in EMBEDDING_CANDIDATES:
        if candidate in lower_to_original:
            col = lower_to_original[candidate]
            vectors = [_parse_vector(v) for v in embedding_df[col]]
            return np.vstack(vectors), str(col)

    # Also support embeddings stored as 384 separate columns.
    indexed = []
    for col in embedding_df.columns:
        match = re.fullmatch(r"(?:embedding|embeddings|dim|feature)_?(\d+)", str(col).lower())
        if match:
            indexed.append((int(match.group(1)), col))
    if len(indexed) >= 32:
        indexed.sort(key=lambda x: x[0])
        cols = [col for _, col in indexed]
        return embedding_df[cols].to_numpy(dtype=np.float32), "separate_dimension_columns"

    # Last-resort detection: a vector-like object column.
    excluded = {"image_id", "observation_uuid", "image_path"}
    for col in embedding_df.columns:
        if str(col).lower() in excluded or len(embedding_df) == 0:
            continue
        sample = embedding_df[col].dropna()
        if len(sample) == 0:
            continue
        first = sample.iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)) or hasattr(first, "as_py"):
            vectors = [_parse_vector(v) for v in embedding_df[col]]
            return np.vstack(vectors), str(col)

    raise ValueError(
        "Could not identify the embedding column. Pass embedding_col explicitly. "
        f"Available columns: {embedding_df.columns.tolist()}"
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("At least one embedding is a zero vector.")
    return matrix / norms


def calculate_visual_neighborhood_features(
    embeddings: np.ndarray,
    k: int = 15,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Calculate local density and visual isolation with cosine nearest neighbors.

    Self-neighbors are removed by index, not merely by assuming the first
    returned neighbor is self. This also handles duplicate embeddings safely.
    """
    X = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    n = len(X)
    if n < 2:
        raise ValueError("At least two images are required for neighborhood features.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    effective_k = min(int(k), n - 1)
    nn = NearestNeighbors(
        n_neighbors=min(effective_k + 1, n),
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    local_density = np.empty(n, dtype=np.float32)
    visual_isolation = np.empty(n, dtype=np.float32)

    for i in range(n):
        keep = indices[i] != i
        row_distances = distances[i][keep][:effective_k]
        if len(row_distances) == 0:
            local_density[i] = np.nan
            visual_isolation[i] = np.nan
        else:
            row_distances = np.clip(row_distances, 0.0, 2.0)
            local_density[i] = float(np.mean(1.0 - row_distances))
            # Isolation is the distance to the closest other image.
            # Local density uses the average similarity across k neighbors,
            # so these two features are not exact mathematical complements.
            visual_isolation[i] = float(np.min(row_distances))

    return pd.DataFrame({
        "local_density": local_density,
        "visual_isolation": visual_isolation,
        "neighborhood_k": effective_k,
    })


def calculate_species_visual_spread(
    embeddings: np.ndarray,
    species_ids: Sequence,
) -> pd.DataFrame:
    """
    For each image, compute cosine distance to its species' normalized centroid.
    Also attaches the species-level mean and standard deviation of that distance.
    """
    X = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    labels = pd.Series(species_ids).reset_index(drop=True)
    if len(labels) != len(X):
        raise ValueError("species_ids length must equal the number of embeddings.")

    result = pd.DataFrame(index=np.arange(len(X)))
    result["species_visual_distance_to_centroid"] = np.nan
    result["species_visual_spread"] = np.nan
    result["species_visual_spread_std"] = np.nan
    result["species_image_count"] = 0

    valid = labels.notna()
    for species_id, positions in labels[valid].groupby(labels[valid]).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        group = X[idx]
        centroid = group.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
            distances = np.clip(1.0 - group @ centroid, 0.0, 2.0)
        else:
            distances = np.full(len(idx), np.nan, dtype=np.float32)

        result.loc[idx, "species_visual_distance_to_centroid"] = distances
        result.loc[idx, "species_visual_spread"] = float(np.nanmean(distances))
        result.loc[idx, "species_visual_spread_std"] = (
            float(np.nanstd(distances, ddof=1)) if len(idx) > 1 else 0.0
        )
        result.loc[idx, "species_image_count"] = len(idx)

    return result.reset_index(drop=True)


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance from one point to arrays of points."""
    earth_radius_km = 6371.0088
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def calculate_species_geographic_diversity(
    observations: pd.DataFrame,
    species_col: str = "species_id",
    observation_id_col: str = "uuid",
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    grid_degrees: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate species-level geographic summaries from unique observations.

    geographic_diversity_entropy is Shannon entropy of occupied degree-grid
    cells (higher means observations are spread across more cells).
    geographic_radius_km is the RMS distance from the spherical mean location.
    """
    required = [species_col, latitude_col, longitude_col]
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise KeyError(f"Missing geographic columns: {missing}")
    if grid_degrees <= 0:
        raise ValueError("grid_degrees must be positive.")

    cols = [species_col, latitude_col, longitude_col]
    if observation_id_col in observations.columns:
        cols.append(observation_id_col)
    data = observations[cols].copy()
    if observation_id_col in data.columns:
        data = data.drop_duplicates(subset=[observation_id_col])
    data[latitude_col] = pd.to_numeric(data[latitude_col], errors="coerce")
    data[longitude_col] = pd.to_numeric(data[longitude_col], errors="coerce")
    data = data.dropna(subset=[species_col, latitude_col, longitude_col])
    data = data[
        data[latitude_col].between(-90, 90)
        & data[longitude_col].between(-180, 180)
    ]

    rows = []
    for species_id, group in data.groupby(species_col):
        lat = group[latitude_col].to_numpy(dtype=float)
        lon = group[longitude_col].to_numpy(dtype=float)

        # Spherical mean avoids the longitude wrap-around problem near +/-180.
        lat_r, lon_r = np.radians(lat), np.radians(lon)
        xyz = np.column_stack((
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ))
        mean_xyz = xyz.mean(axis=0)
        norm = np.linalg.norm(mean_xyz)
        if norm > 0:
            mean_xyz /= norm
            center_lat = np.degrees(np.arctan2(mean_xyz[2], np.hypot(mean_xyz[0], mean_xyz[1])))
            center_lon = np.degrees(np.arctan2(mean_xyz[1], mean_xyz[0]))
            radial = _haversine_km(center_lat, center_lon, lat, lon)
            radius_km = float(np.sqrt(np.mean(radial ** 2)))
        else:
            radius_km = np.nan

        lat_cells = np.floor((lat + 90.0) / grid_degrees).astype(int)
        lon_cells = np.floor((lon + 180.0) / grid_degrees).astype(int)
        cell_codes = pd.Series(list(zip(lat_cells, lon_cells)))
        counts = cell_codes.value_counts().to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        effective_cells = float(np.exp(entropy))

        rows.append({
            species_col: species_id,
            "geographic_observation_count": int(len(group)),
            "geographic_unique_grid_cells": int(len(counts)),
            "geographic_diversity_entropy": entropy,
            "geographic_effective_grid_cells": effective_cells,
            "geographic_radius_km": radius_km,
        })

    return pd.DataFrame(rows)


def calculate_species_temporal_diversity(
    observations: pd.DataFrame,
    species_col: str = "species_id",
    observation_id_col: str = "uuid",
    date_col: str = "observed_on",
) -> pd.DataFrame:
    """Calculate species-level date range, month coverage and month entropy."""
    required = [species_col, date_col]
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise KeyError(f"Missing temporal columns: {missing}")

    cols = [species_col, date_col]
    if observation_id_col in observations.columns:
        cols.append(observation_id_col)
    data = observations[cols].copy()
    if observation_id_col in data.columns:
        data = data.drop_duplicates(subset=[observation_id_col])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce", utc=True)
    data = data.dropna(subset=[species_col, date_col])
    data["__month"] = data[date_col].dt.month
    data["__date"] = data[date_col].dt.date

    rows = []
    for species_id, group in data.groupby(species_col):
        counts = group["__month"].value_counts().to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        rows.append({
            species_col: species_id,
            "temporal_observation_count": int(len(group)),
            "temporal_unique_dates": int(group["__date"].nunique()),
            "temporal_unique_months": int(group["__month"].nunique()),
            "temporal_month_entropy": entropy,
            "temporal_effective_months": float(np.exp(entropy)),
            "temporal_span_days": int((group[date_col].max() - group[date_col].min()).days),
        })

    return pd.DataFrame(rows)


def build_feature_table(
    images_path: str | Path,
    observations_path: str | Path,
    species_path: str | Path,
    embeddings_path: str | Path,
    output_path: Optional[str | Path] = None,
    embedding_col: Optional[str] = None,
    k: int = 15,
    grid_degrees: float = 1.0,
    n_jobs: int = -1,
    include_embedding: bool = False,
) -> pd.DataFrame:
    """
    Load Parquet files, join them, calculate all requested features and return
    one row per image. Optionally writes the result to output_path.
    """
    images = pd.read_parquet(images_path)
    observations = pd.read_parquet(observations_path)
    species = pd.read_parquet(species_path)
    embedding_df = pd.read_parquet(embeddings_path)

    for name, frame, key in (
        ("images", images, "image_id"),
        ("observations", observations, "uuid"),
        ("embeddings", embedding_df, "image_id"),
    ):
        if key not in frame.columns:
            raise KeyError(f"{name} parquet must contain column {key!r}.")

    if "observation_uuid" not in images.columns:
        raise KeyError("images parquet must contain 'observation_uuid'.")
    if "species_id" not in observations.columns:
        raise KeyError("observations parquet must contain 'species_id'.")

    # Ensure one row per image/observation key before merging.
    images = images.drop_duplicates(subset=["image_id"]).copy()
    observations = observations.drop_duplicates(subset=["uuid"]).copy()
    embedding_df = embedding_df.drop_duplicates(subset=["image_id"]).copy()

    matrix, detected_embedding_col = _get_embedding_matrix(embedding_df, embedding_col)
    if len(matrix) != len(embedding_df):
        raise ValueError("Embedding row count does not match embedding table.")
    if matrix.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    embedding_features = pd.DataFrame({
        "image_id": embedding_df["image_id"].to_numpy(),
        "__embedding_row": np.arange(len(embedding_df)),
    })

    # Preserve all image rows that have an embedding; report unmatched rows.
    base = images.merge(embedding_features, on="image_id", how="inner", validate="one_to_one")
    base = base.merge(
        observations,
        left_on="observation_uuid",
        right_on="uuid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_observation"),
    )

    # Add taxonomy without multiplying rows.
    if "species_id" in species.columns:
        species = species.drop_duplicates(subset=["species_id"])
        tax_cols = [c for c in species.columns if c == "species_id" or c not in base.columns]
        base = base.merge(
            species[tax_cols],
            on="species_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_species"),
        )

    row_indices = base["__embedding_row"].to_numpy(dtype=int)
    aligned_matrix = matrix[row_indices]

    visual_features = calculate_visual_neighborhood_features(
        aligned_matrix, k=k, n_jobs=n_jobs
    )
    species_visual = calculate_species_visual_spread(
        aligned_matrix, base["species_id"].to_numpy()
    )

    result = pd.concat(
        [base.reset_index(drop=True), visual_features, species_visual], axis=1
    )

    # Geographic and temporal features are calculated from unique observations,
    # not image rows, so observations with multiple photos are not overweighted.
    geo = calculate_species_geographic_diversity(
        observations, grid_degrees=grid_degrees
    )
    temporal = calculate_species_temporal_diversity(observations)

    if not geo.empty:
        result = result.merge(geo, on="species_id", how="left", validate="many_to_one")
    if not temporal.empty:
        result = result.merge(temporal, on="species_id", how="left", validate="many_to_one")

    result = result.drop(columns=["__embedding_row"], errors="ignore")

    if include_embedding:
        # Store vectors as lists only when explicitly requested.
        result["embedding_vector"] = [row.tolist() for row in aligned_matrix]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)

    # Helpful run metadata for notebook inspection.
    result.attrs["embedding_column_used"] = detected_embedding_col
    result.attrs["embedding_dimensions"] = int(matrix.shape[1])
    result.attrs["input_image_rows"] = int(len(images))
    result.attrs["output_image_rows"] = int(len(result))
    result.attrs["images_without_embedding"] = int(len(images) - len(base))

    return result
