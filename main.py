"""Entry point wiring manifest generation, cleaning, training, evaluation, and visualization."""
import os
import pandas as pd
import torch
import torch.utils.data
import timm

from config import CONFIG, CLASS_MAPPING, KEEP_CLASSES, get_dataset_configs
from data.manifest import DataPreprocessor
from data.validation import DataValidator
from data.postprocessing import DataPostprocessor
from data.dataset import DataStream
from data.split import split_dataset
from data.augmentation import DataAugmentation
from models.classifier import RiceClassifier
from training.trainer import Trainer
from evaluation.metrics import Evaluator
from evaluation.plots import plot_gradcam_multilayer, plot_pca_1024, plot_tsne


def build_manifest():
    dataset_configs = get_dataset_configs(CONFIG["base_dir"])
    preprocessor = DataPreprocessor(dataset_configs, CLASS_MAPPING)
    preprocessor.generate_manifest("/content/manifest.csv")


def validate_and_clean():
    validator = DataValidator("/content/manifest.csv")
    validator.run(output_path="/content/validated_manifest.csv")

    postprocessor = DataPostprocessor("/content/validated_manifest.csv")
    postprocessor.run(keep_classes=KEEP_CLASSES, output_path=CONFIG["manifest_path"])


def verify_paths():
    df = pd.read_csv(CONFIG["manifest_path"])
    missing = df[~df["image_path"].apply(os.path.exists)]
    print(f"Total images: {len(df)}")
    print(f"Missing files: {len(missing)}")
    if len(missing) > 0:
        print("\nSample missing paths:")
        for path in missing["image_path"].head(5):
            print(f"  {path}")
        raise RuntimeError(f"{len(missing)} files missing. Fix dataset extraction before training.")
    else:
        print("All paths verified. Ready to train.")


def train():
    split_dataset(CONFIG["manifest_path"], CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["test_ratio"])

    # shared across every split so label indices can't diverge between them (see DataStream)
    class_to_index = {label: index for index, label in enumerate(sorted(KEEP_CLASSES))}

    dataset_train = DataStream(CONFIG["manifest_path"], split="train", class_to_index=class_to_index)
    augmentation = DataAugmentation(
        manifest=dataset_train.manifest,
        class_to_index=dataset_train.class_to_index,
        backbone_name=CONFIG["backbone_name"],
        input_size=CONFIG["input_size"])
    dataset_train.augmentation = augmentation

    val_transform = timm.data.create_transform(**augmentation.data_config, is_training=False)
    dataset_val = DataStream(CONFIG["manifest_path"], split="val", transform=val_transform, class_to_index=class_to_index)
    dataset_test = DataStream(CONFIG["manifest_path"], split="test", transform=val_transform, class_to_index=class_to_index)

    sampler_train = augmentation.create_sampler()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=CONFIG["batch_size"],
        sampler=sampler_train,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)

    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True)

    model = RiceClassifier(
        backbone_name=CONFIG["backbone_name"],
        num_classes=CONFIG["num_classes"],
        freeze_backbone=CONFIG["freeze_backbone"],
        input_size=CONFIG["input_size"])

    trainer = Trainer(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        criterion=criterion)

    trainer.run(
        stage1_epochs=CONFIG["stage1_epochs"],
        stage2_epochs=CONFIG["stage2_epochs"],
        stage1_lr=CONFIG["stage1_lr"],
        stage2_lr=CONFIG["stage2_lr"],
        weight_decay=CONFIG["weight_decay"],
        checkpoint_path=CONFIG["best_model_path"])

    return model, trainer, dataset_train, dataset_test, loader_test, augmentation


def evaluate(model, loader_test, class_to_index):
    evaluator = Evaluator(model, loader_test, class_to_index)
    evaluator.run(checkpoint_path=CONFIG["best_model_path"])


def visualize(model, dataset_test, loader_test, augmentation):
    index_to_class = {v: k for k, v in dataset_test.class_to_index.items()}

    plot_gradcam_multilayer(
        model, dataset_test, index_to_class,
        num_classes=CONFIG["num_classes"],
        layers_to_show=[20, 25, 31],
        save_path="/content/gradcam_multilayer.png")

    plot_pca_1024(
        model, dataset_test,
        data_config=augmentation.data_config,
        save_path="/content/pca_1024.png")

    plot_tsne(
        model, loader_test, index_to_class,
        num_classes=CONFIG["num_classes"],
        save_path="/content/tsne.png")


if __name__ == "__main__":
    build_manifest()
    validate_and_clean()
    verify_paths()
    model, trainer, dataset_train, dataset_test, loader_test, augmentation = train()
    evaluate(model, loader_test, dataset_train.class_to_index)
    visualize(model, dataset_test, loader_test, augmentation)
