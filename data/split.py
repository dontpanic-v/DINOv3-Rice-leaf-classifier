"""Stratified train/val/test manifest split."""
import pandas as pd
from sklearn.model_selection import train_test_split


# run once before creating DataStream instances — writes the split column back to CSV
def split_dataset(manifest_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    manifest = pd.read_csv(manifest_path)
    labels = manifest["normalized_label"]

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        manifest.index,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state)

    # Second split: val vs test
    temp_labels = labels.iloc[temp_indices]
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels,
        random_state=random_state)

    manifest["split"] = ""
    manifest.loc[train_indices, "split"] = "train"
    manifest.loc[val_indices, "split"] = "val"
    manifest.loc[test_indices, "split"] = "test"

    manifest.to_csv(manifest_path, index=False)

    print(f"Split saved to {manifest_path}")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val: {len(val_indices)}")
    print(f"  Test: {len(test_indices)}")

    return manifest
