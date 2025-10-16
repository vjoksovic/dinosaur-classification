CONFIG = {
  "data": {
    "data_dir": "data",
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1,
    "num_classes": 15,
  },
  "model": {
    "architecture": "custom_cnn",
    "dropout_rate": 0.5,
  },
  "training": {
    "epochs": 50,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
  },
  "scheduler": {
    "type": "cosine", 
    "onecycle": {
      "max_lr": 1e-3,
      "pct_start": 0.3,
      "div_factor": 25.0,
      "final_div_factor": 1e4
    }
  },
  "paths": {
    "model_dir": "models",
    "log_dir": "logs",
    "results_dir": "results",
  },
  "reproducibility": {"seed": 42},
  "logging": {"level": "INFO"},
  "evaluation": {
    "confidence_threshold": 0.5
  },
}


