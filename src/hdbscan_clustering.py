#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image embedding clustering with HDBSCAN.

Run from project root:
    python src/hdbscan_clustering.py --help
    python src/hdbscan_clustering.py --space pca --pca-components 50
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

try:
    import hdbscan
except ImportError as exc:
    raise SystemExit("Missing dependency. Install: pip install hdbscan") from exc


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_embeddings(path: Path, id_col: str, embedding_col: str):
    df = pd.read_parquet(path)
    if id_col not in df.columns:
        raise KeyError(f"ID column {id_col!r} not found. Columns: {list(df.columns)}")
    if embedding_col not in df.columns:
        raise KeyError(f"Embedding column {embedding_col!r} not found. Columns: {list(df.columns)}")
    vectors = np.vstack(df[embedding_col].apply(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix; got shape={vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    return df.reset_index(drop=True), vectors


def build_space(X: np.ndarray, space: str, pca_components: int):
    if space == "original":
        return normalize(X), None
    if space == "pca":
        n_components = min(pca_components, X.shape[1], X.shape[0] - 1)
        model = PCA(n_components=n_components, random_state=42)
        Z = model.fit_transform(X)
        return Z.astype(np.float32), model
    raise ValueError("space must be original or pca")


def fit_hdbscan(Z, min_cluster_size, min_samples, selection):
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=selection,
        prediction_data=True,
        gen_min_span_tree=True,
    )
    labels = model.fit_predict(Z)
    return model, labels


def medoid_distances(Z, labels):
    out = np.full(len(labels), np.nan, dtype=np.float32)
    for label in sorted(set(labels) - {-1}):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 1:
            out[idx] = 0.0
            continue
        # Exact medoid is O(n_cluster^2); cap to a deterministic candidate sample for large clusters.
        candidates = idx if len(idx) <= 1500 else idx[np.linspace(0, len(idx)-1, 1500, dtype=int)]
        block = Z[candidates]
        # Sum squared Euclidean distances using norm expansion.
        norms = np.sum(block * block, axis=1)
        d2 = norms[:, None] + norms[None, :] - 2 * block @ block.T
        medoid_idx = candidates[int(np.argmin(np.maximum(d2, 0).sum(axis=1)))]
        out[idx] = np.linalg.norm(Z[idx] - Z[medoid_idx], axis=1)
    return out


def local_density(Z, k=10):
    n = len(Z)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1).fit(Z)
    distances, _ = nn.kneighbors(Z)
    mean_dist = distances[:, 1:].mean(axis=1)
    return (1.0 / np.maximum(mean_dist, 1e-8)).astype(np.float32)


def evaluate(Z, labels, model):
    non_noise = labels != -1
    n_clusters = len(set(labels) - {-1})
    result = {
        "n_samples": int(len(labels)),
        "n_clusters": int(n_clusters),
        "noise_count": int((~non_noise).sum()),
        "noise_ratio": float((~non_noise).mean()),
        "clustered_ratio": float(non_noise.mean()),
        "cluster_sizes": {str(int(k)): int((labels == k).sum()) for k in sorted(set(labels) - {-1})},
    }
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        # Sample for bounded runtime; excludes noise by definition.
        ids = np.flatnonzero(non_noise)
        if len(ids) > 8000:
            rng = np.random.default_rng(42)
            ids = rng.choice(ids, 8000, replace=False)
        try:
            result["silhouette_non_noise"] = float(silhouette_score(Z[ids], labels[ids], metric="euclidean"))
        except Exception as exc:
            result["silhouette_non_noise_error"] = str(exc)
    try:
        result["relative_validity_dbcv"] = float(model.relative_validity_)
    except Exception:
        result["relative_validity_dbcv"] = None
    return result


def attach_metadata(result_df, root: Path):
    images_path = root / "data/processed/images.parquet"
    obs_path = root / "data/processed/observations.parquet"
    species_path = root / "data/processed/species.parquet"
    if not all(p.exists() for p in [images_path, obs_path, species_path]):
        warnings.warn("Metadata parquet files not all found; saving image-level clustering only.")
        return result_df
    images = pd.read_parquet(images_path)
    observations = pd.read_parquet(obs_path)
    species = pd.read_parquet(species_path)
    needed_img = [c for c in ["image_id", "observation_uuid"] if c in images.columns]
    needed_obs = [c for c in ["uuid", "species_id", "taxon_class_name", "taxon_order_name", "taxon_family_name", "common_name"] if c in observations.columns]
    if not {"image_id", "observation_uuid"}.issubset(images.columns) or "uuid" not in observations.columns:
        warnings.warn("Expected metadata join keys missing; skipping metadata join.")
        return result_df
    meta = images[needed_img].merge(observations[needed_obs], left_on="observation_uuid", right_on="uuid", how="left")
    if "species_id" in meta.columns and "species_id" in species.columns:
        species_cols = [c for c in ["species_id", "common_name", "taxon_class_name", "taxon_order_name", "taxon_family_name"] if c in species.columns]
        sp = species[species_cols].drop_duplicates("species_id")
        # Avoid overwriting observation taxonomy; fill from species table where missing.
        meta = meta.merge(sp, on="species_id", how="left", suffixes=("", "_species"))
        for col in ["common_name", "taxon_class_name", "taxon_order_name", "taxon_family_name"]:
            alt = f"{col}_species"
            if alt in meta.columns:
                if col in meta.columns:
                    meta[col] = meta[col].fillna(meta[alt])
                else:
                    meta[col] = meta[alt]
                meta.drop(columns=alt, inplace=True)
    return result_df.merge(meta.drop(columns=["uuid"], errors="ignore"), on="image_id", how="left")


def add_cluster_taxonomy_metrics(df):
    if "common_name" not in df.columns or "cluster_label" not in df.columns:
        return df, pd.DataFrame()
    rows = []
    for label, group in df[df.cluster_label != -1].groupby("cluster_label"):
        counts = group["common_name"].dropna().value_counts()
        total = counts.sum()
        probs = counts / total if total else pd.Series(dtype=float)
        entropy = float(-(probs * np.log(probs)).sum()) if total else np.nan
        rows.append({
            "cluster_label": int(label),
            "n_images": int(len(group)),
            "n_labeled": int(total),
            "n_species_or_names": int(len(counts)),
            "dominant_name": counts.index[0] if len(counts) else None,
            "taxonomic_purity": float(probs.iloc[0]) if len(probs) else np.nan,
            "species_entropy": entropy,
        })
    metrics = pd.DataFrame(rows)
    return df.merge(metrics[["cluster_label", "taxonomic_purity", "species_entropy"]], on="cluster_label", how="left"), metrics


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/embeddings/images_dinov2.parquet")
    parser.add_argument("--id-col", default="image_id")
    parser.add_argument("--embedding-col", default="embedding")
    parser.add_argument("--space", choices=["original", "pca"], default="pca")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--selection", choices=["eom", "leaf"], default="eom")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    input_path = (root / args.input).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    base, X = load_embeddings(input_path, args.id_col, args.embedding_col)
    Z, reducer = build_space(X, args.space, args.pca_components)
    model, labels = fit_hdbscan(Z, args.min_cluster_size, args.min_samples, args.selection)

    result = base[[args.id_col]].copy()
    result = result.rename(columns={args.id_col: "image_id"})
    result["cluster_label"] = labels
    result["cluster_probability"] = model.probabilities_.astype(np.float32)
    result["outlier_score"] = model.outlier_scores_.astype(np.float32)
    result["distance_to_medoid"] = medoid_distances(Z, labels)
    result["local_density_knn"] = local_density(Z)
    result = attach_metadata(result, root)
    result, cluster_taxonomy = add_cluster_taxonomy_metrics(result)

    result.to_parquet(output_dir / "hdbscan_image_results.parquet", index=False)
    cluster_taxonomy.to_csv(output_dir / "tables/cluster_taxonomy_metrics.csv", index=False)
    metrics = evaluate(Z, labels, model)
    metrics.update({
        "space": args.space,
        "pca_components": args.pca_components if args.space == "pca" else None,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.selection,
        "embedding_shape": list(X.shape),
    })
    (output_dir / "clustering_report.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir / 'hdbscan_image_results.parquet'}")


if __name__ == "__main__":
    main()
