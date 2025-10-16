"""
Grad-CAM visualization script for dinosaur classification.

This script generates Grad-CAM visualizations for sample images from the test set
to understand which parts of the images the model focuses on when making predictions.
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from dataset.dataset_split import load_dataset_split
from model.model import DinoNet
from utils.utils import load_config, set_seed, get_class_names
from utils.grad_cam import GradCAM, save_gradcam_visualization, tensor_to_image


def main():
    """Generate Grad-CAM visualizations for sample test images."""
    
    # Load configuration
    config = load_config("src/config/config.py")
    set_seed(config["reproducibility"]["seed"])
    
    # Get class names
    class_names = get_class_names()
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create results directory
    gradcam_dir = os.path.join(config["paths"]["results_dir"], "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Load test set
    _, _, test_ds = load_dataset_split()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    ckpt_path = os.path.join(config["paths"]["model_dir"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")
    
    model = DinoNet(num_classes=config["data"]["num_classes"], 
                    dropout=config["model"]["dropout_rate"],
                    pretrained=bool(config["model"].get("pretrained", False))).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Initialize Grad-CAM
    # Target the last convolutional block in ResNet-34 backbone
    target_layer = model.backbone.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    
    print(f"Target layer for Grad-CAM: {target_layer}")
    
    # Select random samples for visualization
    num_samples = 10
    total_samples = len(test_ds)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"Generating Grad-CAM visualizations for {len(sample_indices)} samples...")
    
    for i, idx in enumerate(sample_indices):
        # Get the sample
        image, label = test_ds[idx]
        image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension

        # Get prediction
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

        pred_class = pred_class.item()
        true_class = label
        confidence = confidence.item()

        print(f"Sample {i+1}/{len(sample_indices)}: "
              f"True={class_names[true_class]}, "
              f"Pred={class_names[pred_class]}, "
              f"Conf={confidence:.3f}")

        # Generate Grad-CAM heatmap (requires gradients)
        heatmap = gradcam.generate_cam(image_tensor, pred_class)

        # Convert tensor to image for visualization
        original_image = tensor_to_image(image)

        # Create overlay
        overlay = gradcam.overlay_heatmap(original_image, heatmap)

        # Save visualization
        save_path = os.path.join(gradcam_dir, f"gradcam_sample_{i+1:02d}.png")
        save_gradcam_visualization(
            original_image=original_image,
            heatmap=heatmap,
            overlay=overlay,
            predicted_class=class_names[pred_class],
            true_class=class_names[true_class],
            confidence=confidence,
            save_path=save_path
        )
    
    print(f"Grad-CAM visualizations saved to: {gradcam_dir}")
    
    # Generate summary statistics
    print("\nGrad-CAM Summary:")
    print(f"- Generated visualizations for {len(sample_indices)} samples")
    print(f"- Target layer: {target_layer}")
    print(f"- Visualizations saved to: {gradcam_dir}")
    print("- Each visualization shows: original image, heatmap, and overlay")


if __name__ == "__main__":
    main()
