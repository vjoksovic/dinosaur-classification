import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from utils.utils import load_config, set_seed
from dataset.dataset_split import load_dataset_split
from model.model import DinoNet


config = load_config("src/config/config.py")

# Reproducibility and directories
set_seed(config["reproducibility"]["seed"])
os.makedirs(config["paths"]["model_dir"], exist_ok=True)
os.makedirs(config["paths"]["log_dir"], exist_ok=True)

train_ds, val_ds, _ = load_dataset_split()

train_loader = DataLoader(
    train_ds,
    batch_size=config["data"]["batch_size"],
    shuffle=True,
    num_workers=config["data"]["num_workers"],
)
val_loader = DataLoader(
    val_ds,
    batch_size=config["data"]["batch_size"],
    shuffle=False,
    num_workers=config["data"]["num_workers"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DinoNet(
    num_classes=config["data"]["num_classes"],
    dropout=config["model"]["dropout_rate"],
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer_name = config["training"].get("optimizer", "adam").lower()
if optimizer_name == "adam":
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
else:
    optimizer = optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

# Scheduler (configurable)
sched_cfg = config.get("scheduler", {})
sched_type = str(sched_cfg.get("type", "cosine")).lower()
scheduler = None
if sched_type == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"]) 
elif sched_type == "onecycle":
    oc = sched_cfg.get("onecycle", {})
    scheduler = OneCycleLR(
        optimizer,
        max_lr=oc.get("max_lr", config["training"]["learning_rate"]),
        epochs=config["training"]["epochs"],
        steps_per_epoch=0,  # will set after DataLoader is created
        pct_start=oc.get("pct_start", 0.3),
        div_factor=oc.get("div_factor", 25.0),
        final_div_factor=oc.get("final_div_factor", 1e4),
        anneal_strategy="cos",
    )

# AMP scaler
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

best_val_acc = 0.0
best_val_loss = float("inf")
patience = 7
epochs_no_improve = 0

# CSV logger
csv_path = os.path.join(config["paths"]["log_dir"], "training.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

for epoch in range(config["training"]["epochs"]):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # If OneCycleLR, ensure steps_per_epoch is set (first epoch only)
    if isinstance(scheduler, OneCycleLR) and scheduler.total_steps == 0:
        # Recreate OneCycleLR with correct steps_per_epoch
        oc = sched_cfg.get("onecycle", {})
        scheduler = OneCycleLR(
            optimizer,
            max_lr=oc.get("max_lr", config["training"]["learning_rate"]),
            epochs=config["training"]["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=oc.get("pct_start", 0.3),
            div_factor=oc.get("div_factor", 25.0),
            final_div_factor=oc.get("final_div_factor", 1e4),
            anneal_strategy="cos",
        )

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]", 
                      leave=False, unit="batch")
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Per-batch scheduler step for OneCycleLR
        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        
        # Update progress bar
        current_acc = correct / total_samples
        current_loss = total_loss / total_samples
        train_pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })

    train_loss = total_loss / total_samples
    train_acc = correct / total_samples

    # Validation
    model.eval()
    val_total_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]", 
                        leave=False, unit="batch")
        
        for images, labels in val_pbar:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_total_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
            
            # Update progress bar
            current_val_acc = val_correct / val_total
            current_val_loss = val_total_loss / val_total
            val_pbar.set_postfix({
                'loss': f'{current_val_loss:.4f}',
                'acc': f'{current_val_acc:.4f}'
            })

    val_loss = val_total_loss / val_total
    val_acc = val_correct / val_total

    # Per-epoch scheduler step (Cosine)
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()

    # Save best by val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            os.path.join(config["paths"]["model_dir"], "best_model.pth"),
        )

    # Early stopping by val_loss
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Log to CSV
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        # Read current LR safely from optimizer
        current_lr = optimizer.param_groups[0].get("lr", 0.0)
        writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_loss:.6f}",
            f"{val_acc:.6f}",
            f"{current_lr:.8f}",
        ])

    print(
        f"Epoch {epoch+1}/{config['training']['epochs']} - "
        f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
        f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
        f"lr: {optimizer.param_groups[0].get('lr', 0.0):.6f}"
    )

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs (patience={patience}).")
        break
