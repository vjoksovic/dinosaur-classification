# Dinosaur Classification Project

## 🦕 Dataset

The dataset contains images of 15 different dinosaur species:

- Ankylosaurus
- Brachiosaurus
- Compsognathus
- Corythosaurus
- Dilophosaurus
- Dimorphodon
- Gallimimus
- Microceratus
- Pachycephalosaurus
- Parasaurolophus
- Spinosaurus
- Stegosaurus
- Triceratops
- Tyrannosaurus_Rex
- Velociraptor

## 📁 Project Structure

```
dinosaur-classification/
├── src/                      # Source code
│   ├── config/config.py     # Python configuration (CONFIG dict)
│   ├── dataset/
│   │   ├── dino_dataset.py  # Custom Dataset class
│   │   └── dataset_split.py # Train/val/test split logic
│   ├── model/
│   │   └── model.py         # DinoNet CNN (ResNet-34 based)
│   ├── utils/
│   │   ├── transformers.py  # Data augmentation transforms
│   │   ├── utils.py         # Utility functions
│   │   └── grad_cam.py      # Grad-CAM visualization
│   ├── train.py             # Training script with advanced features
│   ├── test.py              # Model evaluation and confusion matrix
│   └── visualize_gradcam.py # Grad-CAM visualization script
├── data/                    # Dataset (15 dinosaur classes)
│   ├── Ankylosaurus/
│   ├── Brachiosaurus/
│   ├── Compsognathus/
│   └── ... (12 more classes)
├── models/                  # Saved model checkpoints
│   └── best_model.pth       # Best model by validation accuracy
├── logs/                    # Training logs
│   └── training.csv         # Training metrics per epoch
├── results/                 # Evaluation results
│   ├── confusion_matrix.png # Confusion matrix visualization
│   └── gradcam/             # Grad-CAM visualizations
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd dinosaur-classification

# Create and activate venv (optional)
python -m venv .venv && .venv\\Scripts\\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 2. Data Preparation

The dataset is already organized in the following structure:
```
data/
├── Ankylosaurus/
├── Brachiosaurus/
├── Compsognathus/
├── Corythosaurus/
├── Dilophosaurus/
├── Dimorphodon/
├── Gallimimus/
├── Microceratus/
├── Pachycephalosaurus/
├── Parasaurolophus/
├── Spinosaurus/
├── Stegosaurus/
├── Triceratops/
├── Tyrannosaurus_Rex/
└── Velociraptor/
```

### 3. Configuration

Configuration je Python dict u `src/config/config.py` (ključ `CONFIG`).
Izmeni vrednosti (batch size, LR, epohe, putanje, split-ovi) po potrebi.

### 4. Training

```bash
# Train the model
python src/train.py
```

### 5. Testing & Evaluation

```bash
# Evaluate model performance and generate confusion matrix
python src/test.py

# Generate Grad-CAM visualizations
python src/visualize_gradcam.py
```

### 6. Available Scripts

- **`src/train.py`**: Main training script with advanced features
- **`src/test.py`**: Model evaluation with confusion matrix generation
- **`src/visualize_gradcam.py`**: Grad-CAM attention visualization

### 7. Checkpoints & Logs

- Best model by validation accuracy is saved to `models/best_model.pth`
- Training metrics per epoch are logged to `logs/training.csv` (train_loss, train_acc, val_loss, val_acc, lr)

## ⚙️ Configuration

`src/config/config.py` contains all hyperparameters and paths:

- **Data**: paths, batch size, image size, train/val/test split ratios
- **Model**: architecture (resnet34), dropout rate, pretrained weights
- **Training**: optimizer (sgd/adamw), learning rate, weight decay, epochs, label smoothing
- **Scheduler**: OneCycleLR or CosineAnnealing with configurable parameters
- **Advanced**: EMA settings, backbone freezing, differential learning rates
- **Paths**: model, log, and results directories
- **Reproducibility**: random seed for consistent results

## 📊 Features

### Training
- **Advanced Learning Rate Scheduling**: OneCycleLR with configurable parameters
- **Early Stopping**: Based on validation loss with configurable patience
- **Model Checkpointing**: Saves best model by validation accuracy
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) on GPU
- **Exponential Moving Average (EMA)**: Optional EMA of model parameters
- **Backbone Freezing**: Configurable warmup epochs with frozen backbone
- **Differential Learning Rates**: Lower LR for pretrained backbone, higher for classifier
- **Label Smoothing**: Configurable label smoothing for better generalization
- **CSV Logging**: Detailed training metrics logged per epoch
- **Reproducibility**: Fixed random seeds for consistent results

### Model Architecture
- **ResNet-34 Backbone**: Pretrained ImageNet weights with custom classifier
- **Dropout Regularization**: Configurable dropout rate in final layers
- **Transfer Learning**: Leverages pretrained ResNet-34 features

### Dataset & Transforms
- **Stratified Split**: Train/val/test split (70/20/10 by default)
- **Advanced Augmentation**: RandomResizedCrop, flips, rotation, color jitter
- **ImageNet Normalization**: Standard preprocessing for pretrained models

### Evaluation & Visualization
- **Comprehensive Testing**: Accuracy, per-class metrics, confusion matrix
- **Grad-CAM Visualization**: Model attention heatmaps for interpretability
- **Confidence Thresholding**: Optional confidence-based prediction filtering
- **Detailed Metrics**: Per-class accuracy and confusion analysis

## 🔧 Usage Examples

### Training the Model

```bash
# Train with default configuration
python src/train.py
```

### Evaluating Model Performance

```bash
# Run comprehensive evaluation
python src/test.py
```

This will:
- Load the best trained model
- Evaluate on test set with confidence thresholding
- Generate confusion matrix visualization
- Show per-class accuracy metrics
- Identify most confused class pairs

### Generating Grad-CAM Visualizations

```bash
# Generate attention heatmaps
python src/visualize_gradcam.py
```

This will:
- Load the trained model
- Select random samples from test set
- Generate Grad-CAM heatmaps showing model attention
- Save visualizations to `results/gradcam/`

### Modifying Configuration

Edit `src/config/config.py` to customize:

```python
CONFIG = {
    "data": {
        "batch_size": 64,        # Increase for faster training
        "train_split": 0.7,      # Adjust data splits
        "val_split": 0.2,
        "test_split": 0.1,
    },
    "model": {
        "dropout_rate": 0.3,     # Adjust regularization
        "pretrained": True,      # Use pretrained weights
    },
    "training": {
        "epochs": 100,           # More training epochs
        "learning_rate": 0.01,   # Adjust learning rate
        "optimizer": "adamw",    # Switch optimizer
    },
    "scheduler": {
        "type": "cosine",        # Use cosine annealing
    }
}
```

## 📈 Monitoring Training

- **CSV Logging**: Training metrics saved to `logs/training.csv`
  - Columns: epoch, train_loss, train_acc, val_loss, val_acc, lr
  - Can be opened in Excel or analyzed with pandas
- **Model Checkpoints**: Best model saved to `models/best_model.pth`
- **Results**: Evaluation results saved to `results/` directory

## 🎯 Performance Tips

1. **Pretrained Weights**: Enable `pretrained: True` in config for better performance
2. **Learning Rate**: Current default is 0.03 with OneCycleLR scheduling
3. **Batch Size**: Increase `batch_size` if you have sufficient GPU memory
4. **EMA**: Enable Exponential Moving Average for more stable training
5. **Backbone Freezing**: Use `freeze_backbone_epochs` for gradual unfreezing
6. **Label Smoothing**: Current default is 0.1 for better generalization
7. **Early Stopping**: Monitor validation loss (patience=7 epochs)


