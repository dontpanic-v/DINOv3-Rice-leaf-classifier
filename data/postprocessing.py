"""Filter a validated manifest down to clean, in-scope images."""
import pandas as pd

from util.images import safe_open_image


class DataPostprocessor:
    """Drops flagged duplicates, corrupted files, low-resolution images, and
    out-of-scope classes from a validated manifest.
    """

    def __init__(self, manifest_path):
        self.manifest = pd.read_csv(manifest_path)

    def _filter_flagged(self):
        before_count = len(self.manifest)
        self.manifest = self.manifest[self.manifest["keep"] == True]
        after_count = len(self.manifest)
        print(f"Filtered flagged: {before_count} -> {after_count} ({before_count - after_count} dropped)")

    def _filter_corrupted(self):
        widths = []
        heights = []
        corrupted_indices = []

        for row in self.manifest.itertuples():
            image = safe_open_image(row.image_path)
            if image is None:
                corrupted_indices.append(row.Index)
                widths.append(0)
                heights.append(0)
            else:
                width, height = image.size
                widths.append(width)
                heights.append(height)

        self.manifest["width"] = widths
        self.manifest["height"] = heights

        before_count = len(self.manifest)
        self.manifest = self.manifest.drop(index=corrupted_indices)
        after_count = len(self.manifest)
        print(f"Filtered corrupted: {before_count} -> {after_count} ({len(corrupted_indices)} dropped)")

    def _filter_low_resolution(self, min_resolution=1024):
        before_count = len(self.manifest)
        resolution_mask = (self.manifest["width"] >= min_resolution) & (self.manifest["height"] >= min_resolution)
        self.manifest = self.manifest[resolution_mask]
        after_count = len(self.manifest)
        print(f"Filtered low resolution (min {min_resolution}x{min_resolution}): {before_count} -> {after_count} ({before_count - after_count} dropped)")

    def _filter_small_classes(self, keep_classes):
        before_count = len(self.manifest)
        before_classes = self.manifest["normalized_label"].nunique()
        self.manifest = self.manifest[self.manifest["normalized_label"].isin(keep_classes)]
        after_count = len(self.manifest)
        after_classes = self.manifest["normalized_label"].nunique()
        print(f"Filtered classes: {before_classes} -> {after_classes} classes, {before_count} -> {after_count} images")

    def _summarize(self):
        total = len(self.manifest)
        total_classes = self.manifest["normalized_label"].nunique()
        total_sources = self.manifest["source"].nunique()

        print(f"\n{'='*60}")
        print(f"Post-processing Summary")
        print(f"{'='*60}")
        print(f"Total images: {total}")
        print(f"Total classes: {total_classes}")
        print(f"Total sources: {total_sources}")

        print(f"\nPer class:")
        class_counts = self.manifest["normalized_label"].value_counts()
        for label, count in class_counts.items():
            print(f"  {label}: {count}")

        print(f"\nPer source:")
        source_counts = self.manifest["source"].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        print(f"\nResolution per dataset:")
        for source in sorted(self.manifest["source"].unique()):
            source_data = self.manifest[self.manifest["source"] == source]
            min_w = source_data["width"].min()
            max_w = source_data["width"].max()
            min_h = source_data["height"].min()
            max_h = source_data["height"].max()
            mean_w = int(source_data["width"].mean())
            mean_h = int(source_data["height"].mean())
            print(f"  {source}: min={min_w}x{min_h}, max={max_w}x{max_h}, mean={mean_w}x{mean_h}")

    def save_final_manifest(self, output_path):
        drop_columns = ["exact_duplicate", "keep"]
        clean_manifest = self.manifest.drop(columns=drop_columns)
        clean_manifest.to_csv(output_path, index=False)
        print(f"\nFinal manifest saved to {output_path}")
        print(f"Total: {len(clean_manifest)} images")

    def run(self, keep_classes, output_path="/content/final_manifest.csv"):
        print("Filtering flagged images...")
        self._filter_flagged()

        print("\nFiltering corrupted images...")
        self._filter_corrupted()

        print("\nFiltering low resolution images...")
        self._filter_low_resolution()

        print("\nFiltering small classes...")
        self._filter_small_classes(keep_classes)

        self._summarize()

        print("\nSaving final manifest...")
        self.save_final_manifest(output_path)
