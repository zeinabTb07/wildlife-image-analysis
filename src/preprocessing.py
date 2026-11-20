
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import imagehash


# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" 

IMAGE_DIR = RAW_DIR / "raw" / "images"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# فایل‌های داده
CSV_FILES = [
    "amphibians.csv",
    "birds.csv",
    "mammals.csv",
    "reptiles.csv",
]

TAXONOMY_COLUMNS = [
    "common_name",
    "taxon_class_name",
    "taxon_order_name",
    "taxon_family_name",
]


def load_observations():
    """خواندن و یکپارچه‌سازی چهار فایل CSV."""

    frames = []

    for filename in CSV_FILES:
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        df["taxon_group"] = filename.replace(".csv", "")
        frames.append(df)

    observations = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    observations["uuid"] = observations["uuid"].astype(str)

    return observations


def calculate_phash(image_path):
    """محاسبه هش ادراکی تصویر."""

    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image.convert("RGB"))
    except Exception:
        return None


def find_duplicate_images(image_paths, threshold=0.90):
    """
    حذف تصاویر با شباهت بیشتر از threshold.

    تصویر اول نگه داشته می‌شود و تصاویر مشابه بعدی
    به‌عنوان تکراری علامت‌گذاری می‌شوند.
    """

    kept_images = []
    removed_images = []
    hashes = {}

    for path in sorted(image_paths):

        current_hash = calculate_phash(path)

        if current_hash is None:
            removed_images.append(path)
            continue

        duplicate = False

        for kept_path in kept_images:
            distance = current_hash - hashes[kept_path]
            similarity = 1 - (distance / 64)

            if similarity > threshold:
                duplicate = True
                removed_images.append(path)
                break

        if not duplicate:
            kept_images.append(path)
            hashes[path] = current_hash

    return kept_images, removed_images


def build_species_table(observations):
    """ساخت جدول گونه‌ها و اختصاص شناسه یکتا."""

    available_columns = [
        col for col in TAXONOMY_COLUMNS
        if col in observations.columns
    ]

    species = (
        observations[available_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    species.insert(
        0,
        "species_id",
        range(1, len(species) + 1)
    )

    return species


def add_species_id(observations, species):
    """اتصال شناسه گونه به جدول مشاهدات."""

    available_columns = [
        col for col in TAXONOMY_COLUMNS
        if col in observations.columns
    ]

    result = observations.merge(
        species,
        on=available_columns,
        how="left"
    )

    return result


def build_images_table(image_paths, observations):
    """ساخت جدول تصاویر و اتصال هر تصویر به مشاهده."""

    observation_uuids = set(observations["uuid"].astype(str))

    rows = []

    for path in image_paths:
        image_uuid = path.stem

        if image_uuid not in observation_uuids:
            continue

        rows.append({
            "image_id": image_uuid,
            "observation_uuid": image_uuid,
            "image_path": str(path.relative_to(BASE_DIR)),
            "animal_present": pd.NA,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "image_id",
            "observation_uuid",
            "image_path",
            "animal_present",
        ]
    )


def build_dataset():
    """اجرای کامل زنجیره پیش‌پردازش."""

    # 1. خواندن داده‌ها
    observations = load_observations()

    # 2. پیدا کردن تصاویر
    image_paths = list(IMAGE_DIR.rglob("*"))

    image_paths = [
        path for path in image_paths
        if path.is_file()
        and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    # 3. حذف تصاویر مشابه
    kept_images, removed_images = find_duplicate_images(
        image_paths,
        threshold=0.90
    )

    # 4. ساخت جدول گونه‌ها
    species = build_species_table(observations)

    # 5. افزودن شناسه گونه به مشاهدات
    observations = add_species_id(observations, species)

    # 6. ساخت جدول تصاویر
    images = build_images_table(kept_images, observations)

    # 7. ذخیره خروجی‌ها
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    observations.to_parquet(
        PROCESSED_DIR / "observations.parquet",
        index=False
    )

    species.to_parquet(
        PROCESSED_DIR / "species.parquet",
        index=False
    )

    images.to_parquet(
        PROCESSED_DIR / "images.parquet",
        index=False
    )

    return observations, species, images, removed_images
