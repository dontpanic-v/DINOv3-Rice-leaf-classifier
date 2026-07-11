"""Perceptual-hash duplicate detection over a manifest CSV."""
import pandas as pd


class DataValidator:
    """Flags exact (phash) duplicates in the manifest and reports per-class duplicate counts."""

    def __init__(self, manifest_path):
        self.manifest = pd.read_csv(manifest_path)

    # duplicated() computed once and reused for both the keep mask and the report
    # (previously computed twice)
    def _flag_exact_duplicates(self):
        is_duplicate = self.manifest.duplicated(subset="phash", keep=False)
        self.manifest["exact_duplicate"] = is_duplicate
        self.manifest["keep"] = ~self.manifest.duplicated(subset="phash", keep="first")
        return self.manifest[is_duplicate].sort_values("phash")

    def generate_report(self, exact_dupes):
        total = len(self.manifest)

        report = {
            "total_images": total,
            "total_classes": self.manifest["normalized_label"].nunique(),
            "exact_duplicates": len(exact_dupes),
            "exact_duplicate_hashes": exact_dupes["phash"].nunique() if len(exact_dupes) > 0 else 0,
            "per_class": {}
        }

        for label in sorted(self.manifest["normalized_label"].unique()):
            class_mask = self.manifest["normalized_label"] == label
            class_total = class_mask.sum()
            class_dupes = len(exact_dupes[exact_dupes["normalized_label"] == label]) if len(exact_dupes) > 0 else 0

            report["per_class"][label] = {
                "total": int(class_total),
                "exact_duplicates": int(class_dupes),
            }

        print(f"\nValidation Report")
        print(f"Total images: {report['total_images']}")
        print(f"Total classes: {report['total_classes']}")
        print(f"Exact duplicates: {report['exact_duplicates']}")
        print(f"\nPer class:")
        for label, stats in report["per_class"].items():
            print(f"  {label}: {stats['total']} total, {stats['exact_duplicates']} dupes")

        return report

    def save_validated_manifest(self, output_path):
        self.manifest.to_csv(output_path, index=False)
        total = len(self.manifest)
        clean = self.manifest["keep"].sum()
        print(f"\nValidated manifest saved to {output_path}")
        print(f"Total: {total}, Keep: {clean}, Remove: {total - clean}")
        print(f"\nClean per class:")
        print(self.manifest[self.manifest["keep"]]["normalized_label"].value_counts().to_string())

    def run(self, output_path="/content/validated_manifest.csv"):
        print("Flagging exact duplicates...")
        exact_dupes = self._flag_exact_duplicates()
        print(f"Exact duplicates: {len(exact_dupes)} images across {exact_dupes['phash'].nunique()} hash groups")
        print(f"Per source:\n{exact_dupes['source'].value_counts().to_string()}")

        print("\nGenerating report...")
        report = self.generate_report(exact_dupes)

        print("\nSaving validated manifest...")
        self.save_validated_manifest(output_path)

        return report
