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
from torchvision import models, transforms


config = load_config("src/config/config.py")

# Reproducibility and directories
set_seed(config["reproducibility"]["seed"])
os.makedirs(config["paths"]["model_dir"], exist_ok=True)
os.makedirs(config["paths"]["log_dir"], exist_ok=True)

# Datasets
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
# Model
model = DinoNet(
    num_classes=config["data"]["num_classes"],
    dropout=config["model"].get("dropout_rate", 0.5),
    pretrained=bool(config["model"].get("pretrained", False)),
).to(device)

# Optionally load pretrained weights if configured
if str(config["model"].get("architecture", "")).lower() == "resnet34":
    use_pretrained = bool(config["model"].get("pretrained", False))
    if use_pretrained:
        try:
            # Rebuild model with pretrained weights
            from model.model import DinoNet as _DinoNet
            import importlib
            importlib.reload(__import__('model.model', fromlist=['DinoNet']))
        except Exception:
            pass

ls = float(config["training"].get("label_smoothing", 0.0))
criterion = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()

optimizer_name = config["training"].get("optimizer", "sgd").lower()
base_lr = float(config["training"].get("learning_rate", 1e-2))
weight_decay = float(config["training"].get("weight_decay", 1e-4))

# Parameter groups: lower LR for backbone, higher for classifier head
backbone_lr_mult = float(config["training"].get("backbone_lr_multiplier", 0.1))
if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
    backbone_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("fc.")]
    head_params = list(model.backbone.fc.parameters())
    param_groups = [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": head_params, "lr": base_lr},
    ]
else:
    param_groups = [{"params": model.parameters(), "lr": base_lr}]

if optimizer_name == "sgd":
    optimizer = optim.SGD(
        param_groups,
        momentum=float(config["training"].get("momentum", 0.9)),
        nesterov=bool(config["training"].get("nesterov", True)),
        weight_decay=weight_decay,
    )
else:
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

# Optionally freeze backbone for warmup epochs
freeze_epochs = int(config["training"].get("freeze_backbone_epochs", 0))
if freeze_epochs > 0 and hasattr(model, "backbone"):
    for n, p in model.backbone.named_parameters():
        if not n.startswith("fc."):
            p.requires_grad = False

# Scheduler (configurable)
sched_cfg = config.get("scheduler", {})
sched_type = str(sched_cfg.get("type", "onecycle")).lower()
scheduler = None
if sched_type == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"]) 
elif sched_type == "onecycle":
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

# AMP scaler
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

best_val_acc = 0.0
best_val_loss = float("inf")
patience = 7
epochs_no_improve = 0

# Exponential Moving Average (EMA) of model params
ema_cfg = config.get("training", {}).get("ema", {})
ema_enabled = bool(ema_cfg.get("enabled", False))
ema_decay = float(ema_cfg.get("decay", 1))
ema_state = None

def _get_model_state_dict(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

if ema_enabled:
    ema_state = _get_model_state_dict(model)

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

    # Unfreeze backbone after warmup epochs and rebuild optimizer with param groups
    if freeze_epochs > 0 and epoch == freeze_epochs:
        if hasattr(model, "backbone"):
            for n, p in model.backbone.named_parameters():
                if not n.startswith("fc."):
                    p.requires_grad = True
            # Rebuild param groups with correct requires_grad flags
            if hasattr(model, "backbone") and hasattr(model.backbone, "fc"):
                backbone_params = [p for n, p in model.backbone.named_parameters() if (not n.startswith("fc.")) and p.requires_grad]
                head_params = [p for p in model.backbone.fc.parameters() if p.requires_grad]
                param_groups = [
                    {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
                    {"params": head_params, "lr": base_lr},
                ]
            else:
                param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]
            if optimizer_name == "sgd":
                optimizer = optim.SGD(
                    param_groups,
                    momentum=float(config["training"].get("momentum", 0.9)),
                    nesterov=bool(config["training"].get("nesterov", True)),
                    weight_decay=weight_decay,
                )
            else:
                optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    # OneCycleLR already initialized with correct steps_per_epoch

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

        # EMA update per step (only for floating-point tensors)
        if ema_enabled:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in ema_state and v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)

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
    swapped_to_ema = False
    if ema_enabled and ema_state is not None:
        # Swap model params with EMA for evaluation
        orig_state = _get_model_state_dict(model)
        # Load only matching floating-point tensors
        safe_ema = {k: v for k, v in ema_state.items() if k in orig_state and v.dtype.is_floating_point}
        model.load_state_dict({**orig_state, **safe_ema}, strict=False)
        swapped_to_ema = True
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

    # Restore original weights after EMA eval
    if swapped_to_ema:
        model.load_state_dict(orig_state, strict=False)

    # Per-epoch scheduler step (Cosine)
    if isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()

    # Save best by val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
         if ema_enabled and ema_state is not None:
            torch.save(ema_state, os.path.join(config["paths"]["model_dir"], "best_model.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(config["paths"]["model_dir"], "best_model.pth"))

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
