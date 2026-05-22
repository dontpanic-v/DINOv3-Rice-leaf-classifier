# Rice Leaf Disease Classification

DINOv3 ViT-Huge+ transfer learning for multi-class rice leaf disease detection. Built for Google Colab with high-end NVIDIA RTX PRO 6000 Blackwell.

## Pipeline Overview

1. **Manifest generation** — crawl 4 Kaggle datasets, normalize labels across 10 naming variants, compute perceptual hashes (phash) for every image.
2. **Data validation** — flag exact duplicates via phash, remove corrupted files, filter images below 1024x1024, drop classes outside the target 8.
3. **Stratified split** — 80/10/10 train/val/test with sklearn `train_test_split`, stratified by class.
4. **Class-aware augmentation** — three transform tiers (light/moderate/heavy) assigned by class frequency. `WeightedRandomSampler` oversamples minority classes.
5. **Two-stage training** — Stage 1: frozen backbone, train BN + MLP head. Stage 2: unfreeze backbone at 1/10th learning rate. bf16 autocast, gradient clipping at 1.0, cosine annealing LR.
6. **Evaluation** — per-class precision/recall/F1, confusion matrix heatmap.
7. **Interpretability** — multi-layer Grad-CAM (blocks 20/25/31), PCA on patch tokens at 1024x1024, t-SNE on CLS embeddings.

## Classes (8)

Bacterial Leaf Blight, Blast, Brown Spot, Healthy, Hispa, Leaf Scald, Narrow Brown Spot, Sheath Blight.

## Project Structure

```
rice-leaf-disease/
    config.py                   # Hyperparameters, paths, class mapping, dataset configs
    main.py                     # Entry point — wires all stages together
    requirements.txt
    data/
        manifest.py             # DataPreprocessor: crawl datasets, build raw CSV
        validation.py           # DataValidator: phash dedup
        postprocessing.py       # DataPostprocessor: filter corrupted/low-res/unwanted classes
        dataset.py              # DataStream: PyTorch Dataset over the manifest
        split.py                # split_dataset: stratified train/val/test split
        transforms.py           # tier-based transform builder (light/moderate/heavy)
        augmentation.py         # DataAugmentation: class weighting + WeightedRandomSampler + tier transform lookup
    models/
        classifier.py           # RiceClassifier: DINOv3 backbone + CLS/patch concat + MLP head
    training/
        engine.py               # train_one_epoch, validate
        trainer.py               # Trainer: two-stage loop, optimizer/scheduler builders, training curves, checkpointing
    evaluation/
        metrics.py              # Evaluator: test inference, classification report, confusion matrix
        gradcam.py              # GradCAM: hook-based class activation maps on ViT blocks (hooks released via remove()/context manager)
        plots.py                 # multi-layer Grad-CAM grid, PCA on patch tokens, t-SNE on CLS embeddings
    util/
        images.py               # list_image_files, safe_open_image, compute_phash
```

## Model Architecture

- **Backbone**: `vit_huge_plus_patch16_dinov3.lvd1689m` from timm (pre-trained, self-supervised)
- **Head input**: CLS token concatenated with mean-pooled patch tokens (embed_dim * 2)
- **Head**: BatchNorm1d -> Linear -> GELU -> Dropout(0.2) -> Linear -> GELU -> Dropout(0.2) -> Linear
- **Prefix tokens**: 5 (1 CLS + 4 register tokens, skipped during patch mean pooling)

## Datasets

All sourced from Kaggle:

| Dataset                          | Kaggle slug                                       |
| -------------------------------- | ------------------------------------------------- |
| RiceLeafs + LabelledRice         | `shayanriyaz/riceleafs`                           |
| RiceDiseaseDataset               | `minhhuy2810/rice-diseases-image-dataset`         |
| RiceLeafsDisease                 | `dedeikhsandwisaputra/rice-leafs-disease-dataset` |
| Rice_Leaf_Diease | `loki4514/rice-leaf-diseases-detection`           |
|                                  |                                                   |

## Usage

On Google Colab, download datasets first (Kaggle API required), then:

```python
from config import CONFIG
from main import build_manifest, validate_and_clean, verify_paths, train, evaluate, visualize

build_manifest()
validate_and_clean()
verify_paths()
model, trainer, train_dataset, test_dataset, test_loader, augmentation = train()
evaluate(model, test_loader, train_dataset.class_to_index)
visualize(model, test_dataset, test_loader, augmentation)
```

Or run all stages:

```bash
python main.py
```

## Key Design Decisions

**Why CLS + patch mean instead of CLS alone?**
DINOv3 distributes spatial information across patch tokens. Averaging patches and concatenating with CLS gives the head both global semantics and local texture signal. This follows the paper's recommendation (Section 6.2.1).

**Why three augmentation tiers?**
Classes range from ~200 to ~2000+ images. Applying uniform heavy augmentation to majority classes wastes compute and can degrade features. The tier system applies augmentation intensity proportional to how much the class needs it.

**Why bf16 instead of fp16?**
DINOv3 was pre-trained in bf16. Using fp16 with GradScaler introduces unnecessary numerical mismatches. bf16 does not require a GradScaler.

**Why two-stage training?**
Stage 1 trains only the head on frozen features, establishing a good classifier before the backbone moves. Stage 2 fine-tunes the backbone at 1/10th learning rate to avoid catastrophic forgetting of pre-trained representations.

## Requirements

```
torch
torchvision
timm
imagehash
pandas
numpy
Pillow
scikit-learn
seaborn
matplotlib
```
