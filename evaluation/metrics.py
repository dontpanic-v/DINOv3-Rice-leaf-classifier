"""Test-set evaluation: accuracy, classification report, confusion matrix."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """Runs test-set inference and reports accuracy, a classification report, and a confusion matrix."""

    def __init__(self, model, test_loader, class_to_index, device="cuda"):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_to_index = class_to_index
        self.index_to_class = {index: label for label, index in class_to_index.items()}
        self.device = device

    def _load_best_model(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")

    def _collect_predictions(self):
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                logits = self.model(images)
                predictions = logits.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_predictions)

    def _print_classification_report(self, true_labels, predicted_labels):
        class_names = [self.index_to_class[i] for i in range(len(self.class_to_index))]
        report = classification_report(
            true_labels, predicted_labels,
            target_names=class_names,
            digits=4)
        print("\nClassification Report:")
        print(report)

    def _plot_confusion_matrix(self, true_labels, predicted_labels):
        class_names = [self.index_to_class[i] for i in range(len(self.class_to_index))]
        conf = confusion_matrix(true_labels, predicted_labels)

        figure = plt.figure(figsize=(8, 6), dpi=100)
        sns.heatmap(
            conf,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
        plt.xlabel("Predicted", fontsize=10)
        plt.ylabel("Actual", fontsize=10)
        plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
        plt.xticks(fontsize=7, rotation=45, ha="right")
        plt.yticks(fontsize=7, rotation=0)
        plt.tight_layout()
        plt.show()
        plt.close(figure)

    def run(self, checkpoint_path):
        print("Loading best model...")
        self._load_best_model(checkpoint_path)

        print("\nRunning inference on test set...")
        true_labels, predicted_labels = self._collect_predictions()

        overall_accuracy = (true_labels == predicted_labels).mean()
        print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")

        self._print_classification_report(true_labels, predicted_labels)

        print("\nPlotting confusion matrix...")
        self._plot_confusion_matrix(true_labels, predicted_labels)

        return overall_accuracy
