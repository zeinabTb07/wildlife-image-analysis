
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "code" / "cleaned_inaturalist"

AUDIT_DIR = (
    BASE_DIR / "code" / "outputs" / "data_audit"
)

DUPLICATE_FILE = (
    AUDIT_DIR / "tables" / "exact_duplicate_images.csv"
)

OUTPUT_DIR = BASE_DIR / "deduplicated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------
# LOAD EXISTING DUPLICATE REPORT
# --------------------------------------------------

duplicates = pd.read_csv(DUPLICATE_FILE)

# گروه‌بندی بر اساس hash؛ نیازی به محاسبه مجدد نیست
duplicate_groups = {
    h: group.copy()
    for h, group in duplicates.groupby("hash")
}

# گروه‌هایی که در بیش از یک کلاس حضور دارند
cross_class_groups = {
    h: group
    for h, group in duplicate_groups.items()
    if group["category"].nunique() > 1
}

# --------------------------------------------------
# LOAD ORIGINAL CSV FILES
# --------------------------------------------------

classes = ["mammals", "birds", "reptiles", "amphibians"]

datasets = {}

for category in classes:
    file_path = DATA_DIR / f"{category}.csv"
    datasets[category] = pd.read_csv(file_path, dtype={"uuid": str})

# UUIDهایی که باید حذف شوند
remove_uuids = set()

# --------------------------------------------------
# HANDLE DUPLICATES WITHIN SAME CLASS
# --------------------------------------------------

print("=" * 60)
print("REMOVING DUPLICATES WITHIN EACH CLASS")
print("=" * 60)

for h, group in duplicate_groups.items():

    # گروه‌های بین‌کلاسی را بعداً دستی بررسی می‌کنیم
    if h in cross_class_groups:
        continue

    rows = group.sort_values("uuid")
    keep_uuid = rows.iloc[0]["uuid"]

    for uuid in rows["uuid"]:
        if uuid != keep_uuid:
            remove_uuids.add(uuid)

    print(
        f"Group: {h[:12]} | "
        f"Keeping: {keep_uuid} | "
        f"Removing: {len(rows) - 1}"
    )

# --------------------------------------------------
# MANUAL CROSS-CLASS DECISIONS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CROSS-CLASS DUPLICATE GROUPS")
print("=" * 60)

manual_decisions = {}

for index, (h, group) in enumerate(cross_class_groups.items(), start=1):

    group = group.sort_values(["category", "uuid"])

    print(f"\nGroup {index}")
    print(f"Hash: {h}")

    for i, row in enumerate(group.itertuples(), start=1):
        print(
            f"  {i}. Category: {row.category} | "
            f"UUID: {row.uuid}"
        )

    print("\nChoose the record to KEEP.")
    print("Enter the number beside the desired UUID.")
    print("Enter 0 to keep ALL records in this group.")

    while True:
        choice = input("Your choice: ").strip()

        if choice.isdigit() and 0 <= int(choice) <= len(group):
            choice = int(choice)
            break

        print("Invalid choice. Try again.")

    if choice == 0:
        print("Keeping all records in this group.")
        continue

    keep_uuid = group.iloc[choice - 1]["uuid"]

    for uuid in group["uuid"]:
        if uuid != keep_uuid:
            remove_uuids.add(uuid)

    manual_decisions[h] = keep_uuid

# --------------------------------------------------
# CREATE DEDUPLICATED DATASETS
# --------------------------------------------------

removed_records = []

for category, df in datasets.items():

    mask = df["uuid"].isin(remove_uuids)

    removed_records.append(df[mask].assign(dataset_class=category))

    clean_df = df[~mask].copy()

    output_file = OUTPUT_DIR / f"{category}.csv"
    clean_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(
        f"\n{category}: "
        f"{len(df)} -> {len(clean_df)} "
        f"({mask.sum()} removed)"
    )

# --------------------------------------------------
# SAVE REMOVED RECORDS
# --------------------------------------------------

if removed_records:
    removed_df = pd.concat(removed_records, ignore_index=True)
else:
    removed_df = pd.DataFrame()

removed_df.to_csv(
    OUTPUT_DIR / "removed_duplicates.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("DEDUPLICATION COMPLETED")
print("=" * 60)

print(f"Removed records: {len(removed_df)}")
print(f"Manually reviewed cross-class groups: {len(manual_decisions)}")
print(f"Output directory: {OUTPUT_DIR}")
