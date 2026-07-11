"""Two-stage (frozen backbone, then fine-tune) training driver."""
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.engine import train_one_epoch, validate


# stage 1: only trainable (head + bn) params — backbone is frozen
def _build_stage1_optimizer(model, learning_rate, weight_decay):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


# stage 2: backbone at 1/10th lr, head + bn at full lr
def _build_stage2_optimizer(model, learning_rate, weight_decay):
    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(map(id, backbone_params))
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    param_groups = [
        {"params": backbone_params, "lr": learning_rate / 10},
        {"params": head_params, "lr": learning_rate}
    ]
    return AdamW(param_groups, weight_decay=weight_decay)


def _plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Val")
    ax_loss.set_xlabel("Epoch", fontsize=8)
    ax_loss.set_ylabel("Loss", fontsize=8)
    ax_loss.set_title("Loss", fontsize=10)
    ax_loss.legend(fontsize=7)

    ax_accuracy.plot(epochs, history["train_accuracy"], label="Train")
    ax_accuracy.plot(epochs, history["val_accuracy"], label="Val")
    ax_accuracy.set_xlabel("Epoch", fontsize=8)
    ax_accuracy.set_ylabel("Accuracy", fontsize=8)
    ax_accuracy.set_title("Accuracy", fontsize=10)
    ax_accuracy.legend(fontsize=7)

    plt.suptitle("Training Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(figure)


class Trainer:
    """Runs stage 1 (frozen backbone) then stage 2 (fine-tune) training, tracking
    per-epoch history and checkpointing on validation-accuracy improvement.
    """

    def __init__(self, model, train_loader, val_loader, criterion, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.best_val_accuracy = 0.0
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def _record(self, train_loss, train_accuracy, val_loss, val_accuracy):
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)

    # Shared epoch loop for both stages — train, validate, step scheduler,
    # log, checkpoint on improvement. Stage 1 and stage 2 previously
    # duplicated this whole loop; they now differ only in which
    # optimizer/scheduler gets built and passed in.
    def _run_stage(self, stage_name, epochs, optimizer, scheduler, checkpoint_path):
        print(f"{stage_name}...")
        for epoch in range(epochs):
            train_loss, train_accuracy = train_one_epoch(
                self.model, self.train_loader, self.criterion, optimizer, self.device)
            val_loss, val_accuracy = validate(
                self.model, self.val_loader, self.criterion, self.device)
            scheduler.step()

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")

            self._record(train_loss, train_accuracy, val_loss, val_accuracy)

            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  Saved best model (val_accuracy: {val_accuracy:.4f})")

    def train_stage1(self, epochs, learning_rate, weight_decay, checkpoint_path):
        optimizer = _build_stage1_optimizer(self.model, learning_rate, weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        self._run_stage("Stage 1: Frozen Backbone", epochs, optimizer, scheduler, checkpoint_path)

    def train_stage2(self, epochs, learning_rate, weight_decay, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model._unfreeze_backbone()
        optimizer = _build_stage2_optimizer(self.model, learning_rate, weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        self._run_stage("\nStage 2: Fine-tune Backbone", epochs, optimizer, scheduler, checkpoint_path)

    def run(self, stage1_epochs, stage2_epochs, stage1_lr, stage2_lr, weight_decay, checkpoint_path):
        self.train_stage1(stage1_epochs, stage1_lr, weight_decay, checkpoint_path)
        self.train_stage2(stage2_epochs, stage2_lr, weight_decay, checkpoint_path)
        _plot_training_curves(self.history)
        print(f"\nTraining complete. Best val accuracy: {self.best_val_accuracy:.4f}")
