import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.images import list_image_files, compute_phash, safe_open_image


def build_manifest(base_dir, dataset_sources, class_mapping, output_path):
    records = []

    for source in dataset_sources:
        root = os.path.join(base_dir, source["root"])
        name = source["name"]

        for dirpath, _, _ in os.walk(root):
            label = os.path.basename(dirpath)
            if label not in class_mapping:
                continue

            for image_path in list_image_files(dirpath):
                records.append({
                    "image_path": image_path,
                    "original_label": label,
                    "normalized_label": class_mapping[label],
                    "phash": compute_phash(image_path),
                    "source": name,
                })

        print(f"[{name}] {sum(1 for r in records if r['source'] == name)} images")

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nmanifest saved to {output_path}")
    print(f"total: {len(df)}, classes: {df['normalized_label'].nunique()}")
    print(df["normalized_label"].value_counts().to_string())
    return df


def deduplicate_manifest(manifest_path, output_path):
    df = pd.read_csv(manifest_path)

    is_dup = df.duplicated(subset="phash", keep=False)
    df["keep"] = ~df.duplicated(subset="phash", keep="first")

    dupes = df[is_dup]
    print(f"duplicates: {len(dupes)} images across {dupes['phash'].nunique()} hash groups")
    print(f"per source:\n{dupes['source'].value_counts().to_string()}")

    df.to_csv(output_path, index=False)

    clean = df["keep"].sum()
    print(f"\nsaved to {output_path}")
    print(f"total: {len(df)}, keep: {clean}, remove: {len(df) - clean}")


def filter_manifest(manifest_path, keep_classes, output_path, min_resolution=1024):
    df = pd.read_csv(manifest_path)

    # drop flagged duplicates
    before = len(df)
    df = df[df["keep"] == True]
    print(f"filtered flagged: {before} -> {len(df)}")

    # drop corrupted files and record dimensions
    widths, heights, corrupted = [], [], []
    for row in df.itertuples():
        image = safe_open_image(row.image_path)
        if image is None:
            corrupted.append(row.Index)
            widths.append(0)
            heights.append(0)
        else:
            w, h = image.size
            widths.append(w)
            heights.append(h)

    df["width"] = widths
    df["height"] = heights
    before = len(df)
    df = df.drop(index=corrupted)
    print(f"filtered corrupted: {before} -> {len(df)}")

    # drop below minimum resolution
    before = len(df)
    mask = (df["width"] >= min_resolution) & (df["height"] >= min_resolution)
    df = df[mask]
    print(f"filtered low-res (min {min_resolution}x{min_resolution}): {before} -> {len(df)}")

    # keep only target classes
    before_classes = df["normalized_label"].nunique()
    df = df[df["normalized_label"].isin(keep_classes)]
    print(f"filtered classes: {before_classes} -> {df['normalized_label'].nunique()}")

    df = df.drop(columns=["exact_duplicate", "keep"], errors="ignore")
    df.to_csv(output_path, index=False)
    print(f"\nsaved to {output_path}")
    print(f"total: {len(df)}")
    print(df["normalized_label"].value_counts().to_string())


def split_manifest(manifest_path, train_ratio, val_ratio, test_ratio, seed=42):
    df = pd.read_csv(manifest_path)
    labels = df["normalized_label"]

    train_idx, temp_idx = train_test_split(
        df.index, test_size=(val_ratio + test_ratio),
        stratify=labels, random_state=seed)

    val_ratio_adj = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio_adj),
        stratify=labels.iloc[temp_idx], random_state=seed)

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    df.to_csv(manifest_path, index=False)
    print(f"split saved to {manifest_path}")
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
