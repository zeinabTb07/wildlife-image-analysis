
from pathlib import Path
import hashlib
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "cleaned_inaturalist"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs" / "data_audit"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

SAMPLE_SIZE = 20
TOP_N = 20
RANDOM_SEED = 42

CLASSES = ["mammals", "birds", "reptiles", "amphibians"]

PERSIAN_NAMES = {
    "mammals": "پستانداران",
    "birds": "پرندگان",
    "reptiles": "خزندگان",
    "amphibians": "دوزیستان"
}

random.seed(RANDOM_SEED)

for folder in [OUTPUT_DIR, TABLE_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

def load_data():

    frames = []

    for category in CLASSES:

        path = DATA_DIR / f"{category}.csv"

        if not path.exists():
            print(f"WARNING: Missing file: {path}")
            continue

        df = pd.read_csv(path)

        df["dataset_class"] = category

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No CSV files found!")

    return pd.concat(frames, ignore_index=True)


# ==================================================
# BASIC DATA AUDIT
# ==================================================

def basic_audit(df):

    print("\n" + "=" * 65)
    print("1. BASIC DATA AUDIT")
    print("=" * 65)

    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = df.isna().sum().to_frame("missing_count")
    missing["percentage"] = missing["missing_count"] / len(df) * 100
    print(missing.to_string())

    missing.to_csv(TABLE_DIR / "missing_values.csv")

    # Unique identifiers
    for col in ["id", "uuid", "user_id"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"Unique: {df[col].nunique():,}")
            print(f"Duplicated: {df[col].duplicated().sum():,}")

    # Save duplicate records
    for col in ["id", "uuid"]:
        if col in df.columns:
            duplicated = df[df.duplicated(col, keep=False)]

            if not duplicated.empty:
                duplicated.to_csv(
                    TABLE_DIR / f"duplicate_{col}.csv",
                    index=False
                )

    structure = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "unique": [df[c].nunique() for c in df.columns]
    })

    structure.to_csv(
        TABLE_DIR / "data_structure.csv",
        index=False
    )


# ==================================================
# CLASS FREQUENCIES
# ==================================================

def class_analysis(df):

    print("\n" + "=" * 65)
    print("2. CLASS FREQUENCIES")
    print("=" * 65)

    counts = df["dataset_class"].value_counts()

    result = pd.DataFrame({
        "count": counts,
        "percentage": counts / len(df) * 100
    })

    print(result.to_string())

    result.to_csv(TABLE_DIR / "class_frequencies.csv")

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")

    plt.title("Observation Count by Class")
    plt.xlabel("Class")
    plt.ylabel("Number of observations")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "class_frequencies.png", dpi=150)
    plt.close()


# ==================================================
# TAXON ANALYSIS
# ==================================================

def taxon_analysis(df):

    print("\n" + "=" * 65)
    print("3. TAXON FREQUENCIES")
    print("=" * 65)

    columns = [
        "common_name",
        "taxon_order_name",
        "taxon_family_name"
    ]

    for col in columns:

        if col not in df.columns:
            continue

        counts = df[col].fillna("Unknown").value_counts()

        print(f"\nTop {TOP_N}: {col}")
        print(counts.head(TOP_N).to_string())

        counts.head(TOP_N).to_csv(
            TABLE_DIR / f"top_{col}.csv"
        )

        counts.to_frame("count").to_csv(
            TABLE_DIR / f"all_{col}_frequencies.csv"
        )

    # Most frequent species/names within each class
    for category in CLASSES:

        subset = df[df["dataset_class"] == category]

        counts = subset["common_name"].fillna(
            "Unknown"
        ).value_counts()

        counts.head(TOP_N).to_csv(
            TABLE_DIR / f"top_taxa_{category}.csv"
        )


# ==================================================
# TIME ANALYSIS
# ==================================================

def time_analysis(df):

    print("\n" + "=" * 65)
    print("4. TEMPORAL ANALYSIS")
    print("=" * 65)

    df["observed_date"] = pd.to_datetime(
        df["observed_on"],
        errors="coerce"
    )

    df["year"] = df["observed_date"].dt.year
    df["month"] = df["observed_date"].dt.month

    print(
        "Records with invalid/missing dates:",
        df["year"].isna().sum()
    )

    # Yearly counts
    yearly = pd.crosstab(
        df["year"],
        df["dataset_class"]
    ).sort_index()

    yearly["total"] = yearly.sum(axis=1)

    print("\nObservations per year:")
    print(yearly.to_string())

    yearly.to_csv(TABLE_DIR / "yearly_counts.csv")

    # Monthly counts
    monthly = pd.crosstab(
        df["month"],
        df["dataset_class"]
    ).reindex(range(1, 13), fill_value=0)

    monthly.to_csv(TABLE_DIR / "monthly_counts.csv")

    # Yearly chart
    yearly.drop(columns="total").plot(
        kind="line",
        marker="o",
        figsize=(11, 6)
    )

    plt.title("Observations per Year by Class")
    plt.xlabel("Year")
    plt.ylabel("Number of observations")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "yearly_observations.png", dpi=150)
    plt.close()

    # Monthly chart
    monthly.plot(
        kind="line",
        marker="o",
        figsize=(11, 6)
    )

    plt.title("Monthly Observation Distribution")
    plt.xlabel("Month")
    plt.ylabel("Number of observations")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "monthly_observations.png", dpi=150)
    plt.close()

    # Yearly stacked bar chart
    yearly.drop(columns="total").plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6)
    )

    plt.title("Yearly Observations by Class")
    plt.xlabel("Year")
    plt.ylabel("Number of observations")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "yearly_stacked.png", dpi=150)
    plt.close()

    return df


# ==================================================
# USER ANALYSIS
# ==================================================

def user_analysis(df):

    print("\n" + "=" * 65)
    print("5. USER ANALYSIS")
    print("=" * 65)

    if "user_id" not in df.columns:
        print("user_id column not found!")
        return

    valid = df.dropna(subset=["user_id"]).copy()

    # Number of observations per user
    uploads = valid.groupby("user_id").agg(
        uploads=("uuid", "count"),
        classes=("dataset_class", "nunique"),
        taxa=("common_name", "nunique")
    )

    uploads = uploads.sort_values(
        "uploads",
        ascending=False
    )

    print(f"\nUnique users: {valid['user_id'].nunique():,}")

    print(f"\nTop {TOP_N} users:")
    print(uploads.head(TOP_N).to_string())

    uploads.to_csv(TABLE_DIR / "user_statistics.csv")

    uploads.head(TOP_N).to_csv(
        TABLE_DIR / "top_users.csv"
    )

    # Distribution of uploads per user
    user_counts = valid.groupby("user_id").size()

    print("\nUser upload statistics:")
    print(user_counts.describe().to_string())

    plt.figure(figsize=(9, 5))

    plt.hist(
        user_counts,
        bins=40,
        log=True
    )

    plt.title("Distribution of Observations per User")
    plt.xlabel("Number of observations")
    plt.ylabel("Number of users (log scale)")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "user_upload_distribution.png", dpi=150)
    plt.close()

    # Top users chart
    uploads["uploads"].head(TOP_N).sort_values().plot(
        kind="barh",
        figsize=(9, 7)
    )

    plt.title(f"Top {TOP_N} Users by Upload Count")
    plt.xlabel("Number of observations")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "top_users.png", dpi=150)
    plt.close()

    # User contribution by class
    user_class = pd.crosstab(
        valid["user_id"],
        valid["dataset_class"]
    )

    user_class.to_csv(TABLE_DIR / "user_class_matrix.csv")


# ==================================================
# MULTI-USER / SAME OBSERVATION ANALYSIS
# ==================================================

def multi_user_analysis(df):

    print("\n" + "=" * 65)
    print("6. MULTI-USER OBSERVATION ANALYSIS")
    print("=" * 65)

    # Same UUID reported under different user IDs
    if {"uuid", "user_id"}.issubset(df.columns):

        uuid_users = df.groupby("uuid").agg(
            user_count=("user_id", "nunique"),
            users=("user_id", lambda x: ", ".join(
                x.dropna().astype(str).unique()
            )),
            record_count=("id", "count")
        )

        shared = uuid_users[uuid_users["user_count"] > 1]

        print(
            "UUIDs associated with multiple users:",
            len(shared)
        )

        shared.to_csv(
            TABLE_DIR / "same_uuid_multiple_users.csv"
        )

    # Candidate reports of the same taxon,
    # date and approximate location.
    required = [
        "observed_on",
        "common_name",
        "latitude",
        "longitude",
        "user_id"
    ]

    if not all(c in df.columns for c in required):
        return

    candidates = df.dropna(subset=required).copy()

    candidates["date_key"] = candidates["observed_on"].astype(str)

    candidates["lat_key"] = pd.to_numeric(
        candidates["latitude"],
        errors="coerce"
    ).round(3)

    candidates["lon_key"] = pd.to_numeric(
        candidates["longitude"],
        errors="coerce"
    ).round(3)

    candidates = candidates.dropna(
        subset=["lat_key", "lon_key"]
    )

    groups = candidates.groupby(
        [
            "date_key",
            "common_name",
            "lat_key",
            "lon_key"
        ]
    ).agg(
        user_count=("user_id", "nunique"),
        users=("user_id", lambda x: ", ".join(
            x.astype(str).unique()
        )),
        observation_count=("id", "count"),
        uuids=("uuid", lambda x: ", ".join(
            x.dropna().astype(str)
        ))
    ).reset_index()

    groups = groups[groups["user_count"] > 1]

    print(
        "Candidate same-date/same-location reports by multiple users:",
        len(groups)
    )

    groups.to_csv(
        TABLE_DIR / "candidate_multi_user_reports.csv",
        index=False
    )


# ==================================================
# IMAGE DUPLICATE ANALYSIS
# ==================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception:
        return None


def image_duplicate_analysis(df):

    print("\n" + "=" * 65)
    print("7. IMAGE DUPLICATE ANALYSIS")
    print("=" * 65)

    image_records = []

    for category in CLASSES:

        folder = IMAGE_DIR / category

        if not folder.exists():
            continue

        subset = df[df["dataset_class"] == category]

        for uuid in subset["uuid"].dropna().astype(str).unique():

            matches = list(folder.glob(uuid + ".*"))

            if not matches:
                continue

            path = matches[0]

            image_records.append({
                "uuid": uuid,
                "category": category,
                "path": str(path),
                "hash": sha256_file(path)
            })

    image_df = pd.DataFrame(image_records)

    if image_df.empty:
        print("No images found!")
        return

    image_df = image_df.dropna(subset=["hash"])

    duplicate_groups = image_df[
        image_df.duplicated("hash", keep=False)
    ].sort_values("hash")

    print(f"Images hashed: {len(image_df):,}")
    print(
        "Images belonging to exact duplicate groups:",
        len(duplicate_groups)
    )
    print(
        "Number of unique duplicate groups:",
        duplicate_groups["hash"].nunique()
    )

    duplicate_groups.to_csv(
        TABLE_DIR / "exact_duplicate_images.csv",
        index=False
    )

    # Images shared across categories
    cross_class = duplicate_groups.groupby("hash").agg(
        categories=("category", lambda x: ", ".join(x.unique())),
        category_count=("category", "nunique"),
        uuids=("uuid", lambda x: ", ".join(x.unique())),
        copies=("uuid", "count")
    )

    cross_class = cross_class[
        cross_class["category_count"] > 1
    ]

    cross_class.to_csv(
        TABLE_DIR / "cross_class_duplicate_images.csv"
    )

    print(
        "Duplicate image groups across classes:",
        len(cross_class)
    )


# ==================================================
# QUALITY GRADE RELATIONSHIPS
# ==================================================

def quality_analysis(df):

    print("\n" + "=" * 65)
    print("8. QUALITY GRADE ANALYSIS")
    print("=" * 65)

    if "quality_grade" not in df.columns:
        return

    table = pd.crosstab(
        df["dataset_class"],
        df["quality_grade"],
        normalize="index"
    ) * 100

    print(table.round(2).to_string())

    table.to_csv(TABLE_DIR / "quality_by_class.csv")

    table.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 5)
    )

    plt.title("Quality Grade Distribution by Class")
    plt.xlabel("Class")
    plt.ylabel("Percentage")
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "quality_by_class.png", dpi=150)
    plt.close()


# ==================================================
# SPATIAL ANALYSIS
# ==================================================

def spatial_analysis(df):

    print("\n" + "=" * 65)
    print("9. SPATIAL DATA AUDIT")
    print("=" * 65)

    if not {"latitude", "longitude"}.issubset(df.columns):
        return

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    valid = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()

    print(f"Valid coordinates: {len(valid):,}")
    print(f"Invalid or missing: {len(df) - len(valid):,}")

    spatial_summary = valid.groupby("dataset_class").agg(
        observations=("uuid", "count"),
        unique_latitudes=("latitude", "nunique"),
        unique_longitudes=("longitude", "nunique"),
        mean_latitude=("latitude", "mean"),
        mean_longitude=("longitude", "mean")
    )

    print(spatial_summary.to_string())

    spatial_summary.to_csv(
        TABLE_DIR / "spatial_summary.csv"
    )

    plt.figure(figsize=(9, 6))

    for category in CLASSES:

        subset = valid[valid["dataset_class"] == category]

        plt.scatter(
            subset["longitude"],
            subset["latitude"],
            s=3,
            alpha=0.3,
            label=category
        )

    plt.title("Observation Coordinates by Class")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_DIR / "spatial_distribution.png", dpi=150)
    plt.close()


# ==================================================
# SAMPLE RECORDS
# ==================================================

def save_samples(df):

    print("\n" + "=" * 65)
    print("10. RANDOM DATA SAMPLES")
    print("=" * 65)

    samples = []

    for category in CLASSES:

        subset = df[df["dataset_class"] == category]

        if subset.empty:
            continue

        sample = subset.sample(
            n=min(SAMPLE_SIZE, len(subset)),
            random_state=RANDOM_SEED
        )

        samples.append(sample)

    result = pd.concat(samples, ignore_index=True)

    result.to_csv(
        TABLE_DIR / "random_samples.csv",
        index=False
    )

    print(f"Saved {len(result)} sample records.")


# ==================================================
# FINAL REPORT
# ==================================================

def main():

    print("=" * 65)
    print("INATURALIST DATA STRUCTURE AUDIT")
    print("=" * 65)

    df = load_data()

    basic_audit(df)
    class_analysis(df)
    taxon_analysis(df)

    df = time_analysis(df)

    user_analysis(df)
    multi_user_analysis(df)
    image_duplicate_analysis(df)
    quality_analysis(df)
    spatial_analysis(df)
    save_samples(df)

    print("\n" + "=" * 65)
    print("AUDIT COMPLETED")
    print("=" * 65)

    print(f"Total records: {len(df):,}")
    print(f"Total users: {df['user_id'].nunique():,}")
    print(f"Total UUIDs: {df['uuid'].nunique():,}")

    print("\nResults saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
