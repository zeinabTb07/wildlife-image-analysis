"""
PCA helpers for DINOv2 image embeddings.

Expected input:
  data/embeddings/images_dinov2.parquet
  columns: image_path and one vector column (embedding/embeddings/vector/features).

The plotting function returns a Plotly FigureWidget. Clicking a point displays
the corresponding original image in the notebook output area.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display
import ipywidgets as widgets
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


EMBEDDING_CANDIDATES = ("embedding", "embeddings", "vector", "features", "image_embedding")


def _find_embedding_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested:
        if requested not in df.columns:
            raise KeyError(f"Embedding column {requested!r} not found. Columns: {list(df.columns)}")
        return requested

    for name in EMBEDDING_CANDIDATES:
        if name in df.columns:
            return name

    # Fallback: detect a column whose first non-null value is a 1-D vector.
    for col in df.columns:
        value = next((x for x in df[col] if x is not None), None)
        if value is not None:
            try:
                arr = np.asarray(value, dtype=np.float32)
                if arr.ndim == 1 and arr.size > 3:
                    return col
            except (TypeError, ValueError):
                pass

    raise ValueError(
        "Could not detect the embedding column. Pass embedding_col='...' explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def load_embeddings(
    parquet_path: str | Path,
    embedding_col: Optional[str] = None,
    image_path_col: str = "image_path",
    id_col: str = "image_id",
    normalize_vectors: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Load parquet and return (dataframe, embedding_matrix, detected_column)."""
    df = pd.read_parquet(parquet_path)
    if image_path_col not in df.columns:
        raise KeyError(f"Missing image path column: {image_path_col}")

    col = _find_embedding_column(df, embedding_col)
    vectors = []
    valid_rows = []
    expected_dim = None

    for idx, value in enumerate(df[col].tolist()):
        try:
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
            if vec.size == 0 or not np.isfinite(vec).all():
                continue
            if expected_dim is None:
                expected_dim = vec.size
            if vec.size != expected_dim:
                continue
            vectors.append(vec)
            valid_rows.append(idx)
        except (TypeError, ValueError):
            continue

    if not vectors:
        raise ValueError(f"No valid vectors found in column {col!r}")

    clean_df = df.iloc[valid_rows].copy().reset_index(drop=True)
    matrix = np.vstack(vectors).astype(np.float32)

    if normalize_vectors:
        matrix = normalize(matrix, norm="l2")

    print(f"Loaded {len(clean_df):,} vectors | dimension={matrix.shape[1]} | column={col!r}")
    return clean_df, matrix, col


def run_pca(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    """Fit PCA and return a copy of df with PC1..PCn columns plus fitted PCA."""
    if n_components < 2 or n_components > min(embeddings.shape):
        raise ValueError(
            f"n_components must be between 2 and {min(embeddings.shape)}, got {n_components}"
        )

    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    coords = pca.fit_transform(embeddings)

    result = df.copy()
    for i in range(n_components):
        result[f"PC{i+1}"] = coords[:, i]

    explained = pca.explained_variance_ratio_ * 100
    print("Explained variance:")
    for i, value in enumerate(explained, start=1):
        print(f"  PC{i}: {value:.2f}%")
    print(f"  Total: {explained.sum():.2f}%")
    return result, pca



def plot_pca_3d(
    pca_df,
    project_root,
    image_path_col="image_path",
    color_col="animal_present",
    id_col="image_id",
    point_size=4,
    height=750,
):
    from pathlib import Path
    from PIL import Image
    from IPython.display import display
    import ipywidgets as widgets
    import plotly.graph_objects as go

    root = Path(project_root).resolve()

    # Image display area
    image_output = widgets.Output(
        layout=widgets.Layout(
            width="400px",
            height=f"{height}px",
            overflow="auto",
            padding="10px",
        )
    )

    # Prepare hover information
    custom = [
        [str(row[image_path_col]), str(row[id_col])]
        for _, row in pca_df.iterrows()
    ]

    marker = dict(
        size=point_size,
        opacity=0.75,
    )

    # Handle categorical colors
    if color_col and color_col in pca_df.columns:
        categories = pca_df[color_col].fillna("Unknown").astype(str)

        palette = [
            "#636EFA", "#EF553B", "#00CC96",
            "#AB63FA", "#FFA15A", "#19D3F3",
            "#FF6692", "#B6E880", "#FF97FF",
            "#FECB52", "#1C83E1", "#FF6692",
        ]

        unique_categories = categories.unique()

        color_map = {
            category: palette[i % len(palette)]
            for i, category in enumerate(unique_categories)
        }

        marker["color"] = categories.map(color_map).tolist()

    # Create 3D figure
    fig = go.FigureWidget(
        data=[
            go.Scatter3d(
                x=pca_df["PC1"],
                y=pca_df["PC2"],
                z=pca_df["PC3"],
                mode="markers",
                marker=marker,
                customdata=custom,
                hovertemplate=(
                    "ID: %{customdata[1]}<br>"
                    "Path: %{customdata[0]}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="DINOv2 PCA 3D",
        height=height,
        width=900,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
    )

    # Display image when hovering over a point
    def handle_hover(trace, points, state):
        if not points.point_inds:
            return

        idx = points.point_inds[0]

        raw_path = Path(
            str(pca_df.iloc[idx][image_path_col])
        )

        if not raw_path.is_absolute():
            raw_path = root / raw_path

        image_path = raw_path.resolve()

        with image_output:
            image_output.clear_output(wait=True)

            print("Image ID:", pca_df.iloc[idx][id_col])
            print("Image path:", image_path)

            if not image_path.exists():
                print("ERROR: Image file not found!")
                return

            try:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")

                    # Resize image to fit the display panel
                    img.thumbnail((370, 550))

                    display(img.copy())

            except Exception as e:
                print("Image loading error:", e)

    # Register hover event
    fig.data[0].on_hover(handle_hover)

    # Horizontal layout: plot + image
    dashboard = widgets.HBox(
        [fig, image_output],
        layout=widgets.Layout(
            width="100%",
            align_items="center",
        ),
    )

    display(dashboard)

    return fig, image_output
