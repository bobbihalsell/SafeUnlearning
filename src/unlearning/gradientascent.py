import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name, num_classes, model_path, device):
    model = getattr(models, model_name)(pretrained=False)

    if model_name.startswith("mobilenet"):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported yet.")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    return model.to(device)


def get_transform(dataset_name):
    if dataset_name.lower() == "imagenet":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif dataset_name.lower() == "cifar10":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465],
                                 [0.2470, 0.2435, 0.2616])
        ])
    elif dataset_name.lower() == "custom":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)["unlearning"]

    seed = config.get("seed", 42)
    model_name = config["model_name"]
    model_path = config["model_path"]
    dataset_name = config["dataset"]
    data_root = config["data_root"]
    save_path = config["save_path"]
    num_classes = config["num_classes"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_name, num_classes, model_path, device)
    model.train()

    # Prepare data
    transform = get_transform(dataset_name)
    forget_set = datasets.ImageFolder(root=data_root, transform=transform)
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Unlearning 
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            (-loss).backward()  
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Gradient Ascent Loss: {total_loss / len(forget_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Unlearned model saved to: {save_path}")


if __name__ == "__main__":
    main()
