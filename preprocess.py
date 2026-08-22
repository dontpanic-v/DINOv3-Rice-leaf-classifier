import os
import sys
import yaml

from src.preprocessing import build_manifest, deduplicate_manifest, filter_manifest, split_manifest

# usage: python preprocess.py <config.yaml>
config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)

pre = config["preprocessing"]
data = config["data"]

manifest_dir = os.path.dirname(data["manifest_path"])
os.makedirs(manifest_dir, exist_ok=True)
raw_path = os.path.join(manifest_dir, "manifest_raw.csv")
validated_path = os.path.join(manifest_dir, "manifest_validated.csv")

print("building manifest ...")
build_manifest(data["base_dir"], pre["dataset_sources"], pre["class_mapping"], raw_path)

print("\ndeduplicating ...")
deduplicate_manifest(raw_path, validated_path)

print("\nfiltering ...")
filter_manifest(validated_path, pre["keep_classes"], data["manifest_path"])

print("\nsplitting ...")
split_manifest(
    data["manifest_path"],
    data["train_ratio"], data["val_ratio"], data["test_ratio"],
    config["seed"],
)

# verify all paths exist
import pandas as pd
df = pd.read_csv(data["manifest_path"])
missing = df[~df["image_path"].apply(os.path.exists)]
print(f"\ntotal images: {len(df)}")
print(f"missing files: {len(missing)}")
if len(missing) > 0:
    for path in missing["image_path"].head(5):
        print(f"  {path}")
    raise RuntimeError(f"{len(missing)} files missing")
else:
    print("all paths verified. ready to train.")
