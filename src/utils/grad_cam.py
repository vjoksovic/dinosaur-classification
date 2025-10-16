"""
Grad-CAM implementation for dinosaur classification model.

This module provides functionality to generate Grad-CAM visualizations
to understand which parts of the input image the model focuses on when making predictions.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    
    This class hooks into the model's forward pass to capture gradients
    and generate attention heatmaps.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The trained model
            target_layer: The convolutional layer to generate CAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save the activation from the target layer."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save the gradient from the target layer."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Class index to generate CAM for. If None, uses predicted class.
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on the original image.
        
        Args:
            image: Original image (H, W, C) in range [0, 255]
            heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
            alpha: Transparency of the heatmap overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        # Overlay heatmap on image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


def save_gradcam_visualization(original_image: np.ndarray, 
                              heatmap: np.ndarray, 
                              overlay: np.ndarray,
                              predicted_class: str,
                              true_class: str,
                              confidence: float,
                              save_path: str) -> None:
    """
    Save Grad-CAM visualization as a figure with original, heatmap, and overlay.
    
    Args:
        original_image: Original image (H, W, C)
        heatmap: Grad-CAM heatmap (H, W)
        overlay: Image with heatmap overlay (H, W, C)
        predicted_class: Predicted class name
        true_class: True class name
        confidence: Prediction confidence
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add prediction info
    fig.suptitle(f'Predicted: {predicted_class} | True: {true_class} | Confidence: {confidence:.3f}', 
                 fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to image for visualization.
    
    Args:
        tensor: Image tensor (C, H, W) with values in range [0, 1]
        
    Returns:
        Image array (H, W, C) with values in range [0, 255]
    """
    # Denormalize (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    
    return image
