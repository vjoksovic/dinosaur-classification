# Dinosaur Classification CNN

A PyTorch-based Convolutional Neural Network for classifying dinosaur species from images. This project implements a deep learning solution to identify 15 different dinosaur species using computer vision techniques.

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
│   ├── dataset/dino_dataset.py
│   ├── dataset/dataset_split.py
│   ├── model/model.py       # DinoNet CNN
│   ├── utils/transformers.py
│   ├── utils/utils.py
│   ├── train.py             # Training script
│   └── test.py              # Simple test accuracy
├── data/                    # Data directory (class folders)
├── models/                  # Saved checkpoints (best_model.pth)
├── logs/                    # CSV training logs
├── notebooks/               # Notebooks
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

# Install core dependencies
pip install torch torchvision scikit-learn matplotlib seaborn pyyaml
```

### 2. Data Preparation

Ensure your data is organized in the following structure:
```
data/raw/
├── Ankylosaurus/
├── Brachiosaurus/
├── Compsognathus/
└── ... (other classes)
```

### 3. Configuration

Configuration je Python dict u `src/config/config.py` (ključ `CONFIG`).
Izmeni vrednosti (batch size, LR, epohe, putanje, split-ovi) po potrebi.

### 4. Training

```bash
# Train the model
python src/train.py
```

### 5. Testing (accuracy)

```
# Evaluate accuracy on held-out test split
python src/test.py
```

### 6. Checkpoints & Logs

- Najbolji model po val-accuracy se čuva u `models/best_model.pth`.
- Metrike po epohi se upisuju u `logs/training.csv` (train_loss, train_acc, val_loss, val_acc, lr).

## ⚙️ Configuration

`src/config/config.py` sadrži sve hiperparametre i putanje:

- **Data**: putanje, batch size, image size, train/val/test split
- **Model**: `dropout_rate`
- **Training**: optimizer (adam/adamw), learning rate, weight decay, epochs
- **Paths**: `models/`, `logs/`, `results/`
- **Reproducibility**: `seed`

## 📊 Features

### Training
- Learning rate scheduling (CosineAnnealing)
- Early stopping (na osnovu val-loss)
- Model checkpointing (best val-acc)
- Mixed precision (AMP) na GPU
- CSV logging metrika po epohi
- Reproducibilnost (seed)

### Dataset & Transforms
- Stratifikovani split na train/val/test (70/15/15 po default-u)
- Osnovne augmentacije za train

## 🔧 Usage Examples

### Training with Custom Settings

```python
from src.train import train_model
from src.utils import load_config

# Load configuration
config = load_config('config/config.yaml')

# Modify settings
config['training']['epochs'] = 50
config['model']['architecture'] = 'resnet50'

# Start training
train_model(config)
```

### Making Predictions

```python
from src.predict import DinosaurPredictor

# Initialize predictor
predictor = DinosaurPredictor('config/config.yaml', 'models/best_model.pth')

# Predict on single image
result = predictor.predict_image('path/to/dinosaur.jpg')
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")

# Get top-5 predictions
top_predictions = predictor.get_top_predictions('path/to/dinosaur.jpg', top_k=5)
for class_name, confidence in top_predictions:
    print(f"{class_name}: {confidence:.4f}")
```

### Evaluating Model Performance

```python
from src.evaluate import evaluate_model

# Evaluate model
results = evaluate_model('config/config.yaml', 'models/best_model.pth')

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Macro F1-Score: {results['macro_avg_f1']:.4f}")
```

## 📈 Monitoring Training

- CSV log: `logs/training.csv` (možeš otvoriti u Excel-u ili pandas-u)

## 🎯 Performance Tips

1. **Data Augmentation**: Use appropriate augmentation to improve generalization
2. **Learning Rate**: Start with 0.001 and use learning rate scheduling
3. **Batch Size**: Use larger batch sizes if you have sufficient GPU memory
4. **Pretrained Models**: Use pretrained weights for better performance
5. **Early Stopping**: Monitor validation loss to prevent overfitting

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Increase `num_workers` in DataLoader
3. **Poor Performance**: Try different architectures or increase training epochs
4. **Import Errors**: Ensure all dependencies are installed correctly

### Getting Help

- Check the configuration file for proper paths and settings
- Verify data organization matches the expected structure
- Ensure sufficient disk space for logs and model checkpoints

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Computer Vision Best Practices](https://pytorch.org/vision/stable/transforms.html)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Torchvision for pretrained models
- The computer vision community for research and best practices
