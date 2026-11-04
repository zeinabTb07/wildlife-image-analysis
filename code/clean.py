
import pandas as pd
import numpy as np
import uuid
from pathlib import Path

# =========================
# 1. Configuration
# =========================

INPUT_FILE = Path(r"D:\Study\SUT-CS\Books\تحلیل داده\پروژه\phase-1\code\iran-2016-2026-observations.csv")
OUTPUT_DIR = Path(r"D:\Study\SUT-CS\Books\تحلیل داده\پروژه\phase-1\code\cleaned_inaturalist")

OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# Approximate bounding box of Iran
IRAN_BOUNDS = {
    "lat_min": 25.0,
    "lat_max": 39.8,
    "lon_min": 44.0,
    "lon_max": 63.5
}

species_groups = {
    "mammals": "Mammalia",
    "birds": "Aves",
    "reptiles": "Reptilia",
    "amphibians": "Amphibia"
}

# =========================
# 2. Load Dataset
# =========================

df = pd.read_csv(INPUT_FILE)

# Clean column names
df.columns = df.columns.str.strip()

initial_count = len(df)

print("=" * 50)
print("INITIAL DATASET")
print("=" * 50)
print(f"Initial records: {initial_count}")

# Normalize empty strings and whitespace
df = df.replace(r"^\s*$", np.nan, regex=True)

# =========================
# 3. Remove Duplicate IDs
# =========================

df = df.drop_duplicates(subset=["id"]).copy()

# =========================
# 4. Taxon Geoprivacy
# =========================

# Missing taxon_geoprivacy -> open
df["taxon_geoprivacy"] = (
    df["taxon_geoprivacy"]
    .fillna("open")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert coordinates to numeric
df["latitude"] = pd.to_numeric(
    df["latitude"], errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"], errors="coerce"
)

# =========================
# 5. Handle Obscured Locations
# =========================

obscured_mask = (
    df["taxon_geoprivacy"] == "obscured"
)

obscured_count = obscured_mask.sum()

# Generate synthetic coordinates
df.loc[obscured_mask, "latitude"] = np.random.uniform(
    IRAN_BOUNDS["lat_min"],
    IRAN_BOUNDS["lat_max"],
    obscured_count
)

df.loc[obscured_mask, "longitude"] = np.random.uniform(
    IRAN_BOUNDS["lon_min"],
    IRAN_BOUNDS["lon_max"],
    obscured_count
)

# Mark synthetic locations
df["location_is_synthetic"] = False

df.loc[
    obscured_mask, "location_is_synthetic"
] = True

print(f"\nObscured records: {obscured_count}")

# =========================
# 6. Validate Coordinates
# =========================

# Remove invalid coordinates
valid_coordinates = (
    df["latitude"].between(-90, 90) &
    df["longitude"].between(-180, 180)
)

df = df[valid_coordinates].copy()

# Keep coordinates inside approximate Iran bounds
iran_mask = (
    df["latitude"].between(
        IRAN_BOUNDS["lat_min"],
        IRAN_BOUNDS["lat_max"]
    ) &
    df["longitude"].between(
        IRAN_BOUNDS["lon_min"],
        IRAN_BOUNDS["lon_max"]
    )
)

df = df[iran_mask].copy()

# =========================
# 7. Remove Missing Image URLs
# =========================

# Image URL is required
df = df.dropna(subset=["image_url"])

df = df[
    df["image_url"].astype(str).str.strip() != ""
].copy()

# =========================
# 8. Date and Time Processing
# =========================

# Parse observation date
df["observed_on"] = pd.to_datetime(
    df["observed_on"],
    errors="coerce"
)

# Parse observation datetime
df["time_observed_at"] = pd.to_datetime(
    df["time_observed_at"],
    errors="coerce",
    utc=True
)
# =========================
# 9. Fill Missing Dates/Times
#    Using Family Mean
# =========================

# Calculate mean observation date per family

family_mean_dates = (
    df.groupby("taxon_family_name")["observed_on"]
    .transform(
        lambda x: x.dropna().mean()
        if x.notna().any() else pd.NaT
    )
)

# Fill missing observation dates

missing_dates = df["observed_on"].isna()

df.loc[missing_dates, "observed_on"] = (
    family_mean_dates[missing_dates]
)


# Calculate mean observation time per family

family_mean_times = (
    df.groupby("taxon_family_name")["time_observed_at"]
    .transform(
        lambda x: x.dropna().mean()
        if x.notna().any() else pd.NaT
    )
)

# Fill missing observation times

missing_times = df["time_observed_at"].isna()

df.loc[missing_times, "time_observed_at"] = (
    family_mean_times[missing_times]
)


print("\nDate/time imputation:")
print(f"Missing dates filled: {missing_dates.sum()}")
print(f"Missing times filled: {missing_times.sum()}")
# =========================
# 10. Validate UUID
# =========================

def is_valid_uuid(value):
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


df = df[
    df["uuid"].apply(is_valid_uuid)
].copy()

# =========================
# 11. Validate Required Fields
# =========================

required_columns = [
    "id",
    "uuid",
    "observed_on",
    "time_observed_at",
    "user_id",
    "common_name",
    "taxon_class_name",
    "taxon_order_name",
    "taxon_family_name",
    "latitude",
    "longitude",
    "image_url"
]

# Remove records with remaining missing values
df = df.dropna(
    subset=required_columns
).copy()

# Remove empty strings
for col in required_columns:
    if df[col].dtype == "object":
        df = df[
            df[col].astype(str).str.strip() != ""
        ].copy()

# =========================
# 12. Normalize Taxonomy
# =========================

df["taxon_class_name"] = (
    df["taxon_class_name"]
    .astype(str)
    .str.strip()
)

# =========================
# 13. Export Four CSV Files
# =========================

print("\n" + "=" * 50)
print("EXPORT RESULTS")
print("=" * 50)

for filename, class_name in species_groups.items():

    # Initial number of records for this class
    initial_group_count = (
        pd.read_csv(
            INPUT_FILE,
            usecols=["taxon_class_name"]
        )["taxon_class_name"]
        .astype("string")
        .str.strip()
        .eq(class_name)
        .sum()
    )

    # Final cleaned records
    group_df = df[
        df["taxon_class_name"] == class_name
    ].copy()

    final_group_count = len(group_df)

    removed_count = (
        initial_group_count - final_group_count
    )

    # Export
    output_path = OUTPUT_DIR / f"{filename}.csv"

    group_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n{filename.upper()}")
    print(f"Initial rows: {initial_group_count}")
    print(f"Final rows: {final_group_count}")
    print(f"Removed rows: {removed_count}")
    print(f"Saved: {output_path}")

# =========================
# 14. Final Summary
# =========================

print("\n" + "=" * 50)
print("FINAL CLEANING REPORT")
print("=" * 50)

print(f"Original dataset: {initial_count}")
print(f"Final cleaned dataset: {len(df)}")
print(f"Total removed: {initial_count - len(df)}")
print(f"Obscured locations processed: {obscured_count}")

print("\nAll files saved in:", OUTPUT_DIR)
