"""Crawl raw dataset folders/CSVs into a single normalized manifest CSV."""
import os
import pandas as pd

from util.images import list_image_files, compute_phash


class DataPreprocessor:
    """Builds one manifest row per image across all configured datasets,
    normalizing each dataset's own label naming via class_mapping.
    """

    def __init__(self, datasets_config, class_mapping):
        self.datasets_config = datasets_config
        self.class_mapping = class_mapping

    # handles both folder-based and csv-based label sources
    def _process_dataset(self, config):
        records = []
        root = config["root_path"]
        name = config["dataset_name"]
        label_source = config.get("label_source", "folder")

        if label_source == 'csv':
            df = pd.read_csv(config["csv_path"])
            for _, row in df.iterrows():
                filename = row.iloc[0]
                original_label = row.iloc[1]

                if original_label not in self.class_mapping:
                    continue

                images_path = os.path.join(root, filename)
                records.append({
                    "image_path": images_path,
                    "original_label": original_label,
                    "normalized_label": self.class_mapping[original_label],
                    "phash": compute_phash(images_path),
                    "source": name
                })
        else:
            for dirpath, _, _ in os.walk(root):
                label = os.path.basename(dirpath)

                if label not in self.class_mapping:
                    continue

                for image_path in list_image_files(dirpath):
                    records.append({
                        "image_path": image_path,
                        "original_label": label,
                        "normalized_label": self.class_mapping[label],
                        "phash": compute_phash(image_path),
                        "source": name
                    })

        return records

    def generate_manifest(self, output_path):
        all_records = []

        for config in self.datasets_config:
            records = self._process_dataset(config)
            print(f"[{config['dataset_name']}] Collected {len(records)} images")
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        df.to_csv(output_path, index=False)

        print(f"\nManifest saved to {output_path}")
        print(f"Total images: {len(df)}")
        print(f"Classes: {df['normalized_label'].nunique()}")
        print(f"Per class:\n{df['normalized_label'].value_counts().to_string()}")

        return df
