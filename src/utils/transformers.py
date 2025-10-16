from torchvision import transforms

train_transform = transforms.Compose([
    # RandomResizedCrop introduces random cropping and scale/zoom
    transforms.RandomResizedCrop(
        size=(224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.25)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # vertical flip with small probability
    transforms.RandomRotation(15),  # slightly larger rotation range
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
