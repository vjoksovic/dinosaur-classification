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
    "architecture": "resnet34",
    "dropout_rate": 0.5,
    "pretrained": True
  },
  "training": {
    "epochs": 50,
    "optimizer": "sgd",
    "learning_rate": 3e-2,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "nesterov": True,
    "label_smoothing": 0.1,
    "backbone_lr_multiplier": 0.05,
    "freeze_backbone_epochs": 2,
    "ema": {
      "enabled": True,
      "decay": 0.999
    }
  },
  "scheduler": {
    "type": "onecycle", 
    "onecycle": {
      "max_lr": 3e-2,
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


