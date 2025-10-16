import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset_split import load_dataset_split
from model.model import DinoNet
from utils.utils import load_config, set_seed, get_class_names
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

config = load_config("src/config/config.py")
set_seed(config["reproducibility"]["seed"])

# Get class names
class_names = get_class_names()
print(f"Found {len(class_names)} classes: {class_names}")

# Create results directory
os.makedirs(config["paths"]["results_dir"], exist_ok=True)

# Load test set
_, _, test_ds = load_dataset_split()
test_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
ckpt_path = os.path.join(config["paths"]["model_dir"], "best_model.pth")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

model = DinoNet(num_classes=config["data"]["num_classes"], dropout=config["model"]["dropout_rate"]).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

correct = 0
total = 0
num_classes = config["data"]["num_classes"]
threshold = float(config.get("evaluation", {}).get("confidence_threshold", 0.5))

# Per-class counters
per_class_correct = [0 for _ in range(num_classes)]
per_class_total = [0 for _ in range(num_classes)]

# Lists to store all predictions and labels for confusion matrix
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, dim=1)

        # Apply confidence threshold: mark low-confidence as unknown (-1)
        unknown_mask = confs < threshold
        preds_thresholded = preds.clone()
        preds_thresholded[unknown_mask] = -1

        # Only count as correct when not unknown and class matches
        matches = (preds_thresholded == labels)
        matches[unknown_mask] = False
        correct += matches.sum().item()
        total += labels.size(0)

        # Store predictions and labels for confusion matrix
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Per-class stats (ground-truth based)
        for c in range(num_classes):
            class_mask = (labels == c)
            per_class_total[c] += class_mask.sum().item()
            # correct only when predicted c with confidence >= threshold
            per_class_correct[c] += ((preds_thresholded == c) & class_mask).sum().item()

overall_acc = (correct / total) if total > 0 else 0.0
print(f"Test Accuracy (thresholded): {overall_acc:.4f}")

# Per-class accuracy
print("Per-class accuracy:")
for c in range(num_classes):
    denom = per_class_total[c]
    acc_c = (per_class_correct[c] / denom) if denom > 0 else 0.0
    print(f"  {class_names[c]}: {acc_c:.4f}  (n={denom})")

# Generate and save confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(all_labels, all_predictions)

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Dinosaur Classification')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save confusion matrix
cm_path = os.path.join(config["paths"]["results_dir"], "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {cm_path}")
plt.close()

# Find top 5 most confused class pairs
print("\nTop 5 most confused class pairs:")
confusion_pairs = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((cm[i, j], class_names[i], class_names[j]))

confusion_pairs.sort(reverse=True)
for i, (count, true_class, pred_class) in enumerate(confusion_pairs[:5]):
    print(f"  {i+1}. {true_class} → {pred_class}: {count} misclassifications")

# Unknown rate
print("\nUnknown (low-confidence) rate by batch is applied; overall accuracy ignores unknowns as correct.")
