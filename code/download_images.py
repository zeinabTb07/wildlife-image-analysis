
import pandas as pd
import requests
import os
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

# =========================
# 1. Configuration
# =========================

BASE_DIR = Path(
    r"D:\Study\SUT-CS\Books\تحلیل داده\پروژه\phase-1\code"
)

DATA_DIR = BASE_DIR / "cleaned_inaturalist"
IMAGE_DIR = DATA_DIR / "images"

species_groups = [
    "mammals",
    "birds",
    "reptiles",
    "amphibians"
]

# Number of simultaneous downloads
MAX_WORKERS = 6

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

thread_local = local()


# =========================
# 2. HTTP Session
# =========================

def get_session():

    if not hasattr(thread_local, "session"):

        session = requests.Session()

        session.headers.update({
            "User-Agent": "iNaturalist-Image-Downloader/1.0"
        })

        thread_local.session = session

    return thread_local.session


# =========================
# 3. Download Function
# =========================

def download_image(row, folder):

    image_url = row["image_url"]
    image_uuid = str(row["uuid"])

    # Determine image extension
    parsed_url = urlparse(image_url)

    extension = Path(parsed_url.path).suffix.lower()

    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        extension = ".jpg"

    output_path = folder / f"{image_uuid}{extension}"

    # Skip already downloaded images
    if output_path.exists() and output_path.stat().st_size > 0:
        return "skipped", image_uuid

    try:

        session = get_session()

        response = session.get(
            image_url,
            timeout=(10, 30)
        )

        response.raise_for_status()

        # Validate image content type
        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if not content_type.startswith("image/"):
            return "failed", image_uuid

        # Save image
        temp_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        with open(temp_path, "wb") as file:
            file.write(response.content)

        # Rename temporary file after successful download
        os.replace(temp_path, output_path)

        return "success", image_uuid

    except Exception as error:

        return "failed", image_uuid


# =========================
# 4. Process Each Group
# =========================

total_success = 0
total_failed = 0
total_skipped = 0

failed_records = []


for group in species_groups:

    print("\n" + "=" * 50)
    print(f"PROCESSING: {group.upper()}")
    print("=" * 50)

    csv_path = DATA_DIR / f"{group}.csv"

    if not csv_path.exists():

        print(f"CSV not found: {csv_path}")
        continue

    # Read cleaned dataset
    df = pd.read_csv(csv_path)

    # Create image folder
    group_folder = IMAGE_DIR / group
    group_folder.mkdir(parents=True, exist_ok=True)

    print(f"Total records: {len(df)}")

    # Prepare records
    records = df[
        ["uuid", "image_url"]
    ].dropna().to_dict("records")

    success = 0
    failed = 0
    skipped = 0

    # Download images concurrently
    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                download_image,
                row,
                group_folder
            )
            for row in records
        ]

        for i, future in enumerate(
            as_completed(futures), start=1
        ):

            status, image_uuid = future.result()

            if status == "success":
                success += 1

            elif status == "skipped":
                skipped += 1

            else:
                failed += 1

                failed_records.append({
                    "group": group,
                    "uuid": image_uuid
                })

            # Progress report
            if i % 100 == 0 or i == len(futures):

                print(
                    f"Progress: {i}/{len(futures)} | "
                    f"Downloaded: {success} | "
                    f"Skipped: {skipped} | "
                    f"Failed: {failed}"
                )

    total_success += success
    total_failed += failed
    total_skipped += skipped

    print(f"\n{group.upper()} RESULTS")
    print(f"Downloaded: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


# =========================
# 5. Save Failed Records
# =========================

if failed_records:

    failed_df = pd.DataFrame(failed_records)

    failed_df.to_csv(
        IMAGE_DIR / "failed_downloads.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 6. Final Report
# =========================

print("\n" + "=" * 50)
print("FINAL DOWNLOAD REPORT")
print("=" * 50)

print(f"Successfully downloaded: {total_success}")
print(f"Already existed: {total_skipped}")
print(f"Failed downloads: {total_failed}")

print("\nImages saved in:")
print(IMAGE_DIR)

print("\nDownload process completed!")
