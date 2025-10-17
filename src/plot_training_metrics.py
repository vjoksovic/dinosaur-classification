import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the training data
df = pd.read_csv('logs/training.csv')

# Create the plot
plt.figure(figsize=(12, 8))

# Plot training and validation accuracy
plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4)

# Customize the plot
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Set axis limits and ticks
plt.xlim(1, df['epoch'].max())
plt.ylim(0, 1.0)
plt.xticks(range(0, df['epoch'].max() + 1, 5))

# Add some styling
plt.tight_layout()

# Save the plot
plt.savefig('results/training_accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Final Training Accuracy: {df['train_acc'].iloc[-1]:.4f}")
print(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.4f}")
print(f"Best Validation Accuracy: {df['val_acc'].max():.4f} (Epoch {df.loc[df['val_acc'].idxmax(), 'epoch']})")
print(f"Total Epochs: {df['epoch'].max()}")
