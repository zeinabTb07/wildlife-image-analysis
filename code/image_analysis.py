
from pathlib import Path
import random
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat, ImageDraw

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "cleaned_inaturalist"
IMAGE_DIR = DATA_DIR / "images"

OUTPUT_DIR = BASE_DIR / "outputs" / "image_analysis"
SAMPLE_SIZE = 100

CLASSES = {
    "mammals": "پستانداران",
    "birds": "پرندگان",
    "reptiles": "خزندگان",
    "amphibians": "دوزیستان"
}

EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# --------------------------------------------------
# FIND IMAGE
# --------------------------------------------------

def find_image(uuid, folder):
    for ext in EXTENSIONS:
        path = folder / f"{uuid}{ext}"
        if path.exists():
            return path
    return None


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

def analyze_image(path):
    try:
        with Image.open(path) as img:
            img.load()

            width, height = img.size
            brightness = ImageStat.Stat(img.convert("L")).mean[0]

            return {
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height else np.nan,
                "file_size_kb": path.stat().st_size / 1024,
                "brightness": brightness,
                "format": img.format,
                "status": "OK"
            }

    except Exception as e:
        return {
            "width": np.nan,
            "height": np.nan,
            "aspect_ratio": np.nan,
            "file_size_kb": np.nan,
            "brightness": np.nan,
            "format": None,
            "status": f"ERROR: {str(e)}"
        }


# --------------------------------------------------
# CONTACT SHEET
# --------------------------------------------------

def create_contact_sheet(samples, category, output_dir):

    columns = 15
    thumb_size = (160, 130)
    cell_width = 180
    cell_height = 165

    rows = math.ceil(len(samples) / columns)

    if rows == 0:
        print(f"No images available for {category}")
        return

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(samples):

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail(thumb_size)

                x = (i % columns) * cell_width
                y = (i // columns) * cell_height

                image_x = x + (cell_width - img.width) // 2
                image_y = y + 5

                sheet.paste(img, (image_x, image_y))

                uuid = path.stem[:18]
                draw.text(
                    (x + 5, y + 140),
                    uuid,
                    fill="black"
                )

        except Exception:
            continue

    output_path = output_dir / f"{category}_samples.jpg"
    sheet.save(output_path, quality=90)


# --------------------------------------------------
# LOAD AND PROCESS DATA
# --------------------------------------------------

def main():

    random.seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary = []

    print("=" * 60)
    print("INITIAL IMAGE ANALYSIS")
    print("=" * 60)

    for category, persian_name in CLASSES.items():

        print(f"\nProcessing: {persian_name}")
        print("-" * 40)

        csv_path = DATA_DIR / f"{category}.csv"
        image_folder = IMAGE_DIR / category

        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue

        if not image_folder.exists():
            print(f"Image folder not found: {image_folder}")
            continue

        df = pd.read_csv(csv_path)

        if "uuid" not in df.columns:
            print("ERROR: uuid column not found!")
            continue

        uuids = df["uuid"].dropna().astype(str).drop_duplicates()

        records = []
        available_images = []

        missing_count = 0
        error_count = 0

        for uuid in uuids:

            image_path = find_image(uuid, image_folder)

            if image_path is None:
                missing_count += 1
                continue

            result = analyze_image(image_path)

            record = {
                "uuid": uuid,
                "category": category,
                "image_path": str(image_path),
                **result
            }

            records.append(record)

            if result["status"] == "OK":
                available_images.append(image_path)
            else:
                error_count += 1

        all_results.extend(records)

        # Random sample for visual inspection
        sample_count = min(SAMPLE_SIZE, len(available_images))

        samples = random.sample(
            available_images,
            sample_count
        )

        create_contact_sheet(
            samples,
            category,
            OUTPUT_DIR
        )

        # Summary
        valid_records = [
            r for r in records if r["status"] == "OK"
        ]

        summary.append({
            "category": category,
            "total_observations": len(uuids),
            "available_images": len(available_images),
            "missing_images": missing_count,
            "unreadable_images": error_count,
            "sampled_images": sample_count
        })

        print(f"Observations: {len(uuids)}")
        print(f"Available images: {len(available_images)}")
        print(f"Missing images: {missing_count}")
        print(f"Unreadable images: {error_count}")
        print(f"Images selected for contact sheet: {sample_count}")

        if valid_records:
            valid_df = pd.DataFrame(valid_records)

            print(
                f"Average resolution: "
                f"{valid_df['width'].mean():.0f} x "
                f"{valid_df['height'].mean():.0f}"
            )

            print(
                f"Average file size: "
                f"{valid_df['file_size_kb'].mean():.1f} KB"
            )

            print(
                f"Average brightness: "
                f"{valid_df['brightness'].mean():.1f} / 255"
            )

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary)

    results_df.to_csv(
        OUTPUT_DIR / "image_statistics.csv",
        index=False
    )

    summary_df.to_csv(
        OUTPUT_DIR / "image_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    if not results_df.empty:

        valid = results_df[
            results_df["status"] == "OK"
        ].copy()

        if not valid.empty:

            metrics = [
                ("width", "Image width (pixels)"),
                ("height", "Image height (pixels)"),
                ("aspect_ratio", "Aspect ratio"),
                ("file_size_kb", "File size (KB)"),
                ("brightness", "Mean brightness (0-255)")
            ]

            for column, label in metrics:

                plt.figure(figsize=(9, 5))

                data = [
                    valid.loc[
                        valid["category"] == category,
                        column
                    ].dropna()
                    for category in CLASSES
                ]

                plt.boxplot(
                    data,
                    tick_labels=list(CLASSES.keys()),
                    showfliers=False
                )

                plt.title(label + " by category")
                plt.xlabel("Category")
                plt.ylabel(label)
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()

                plt.savefig(
                    OUTPUT_DIR / f"{column}_comparison.png",
                    dpi=150
                )

                plt.close()

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
