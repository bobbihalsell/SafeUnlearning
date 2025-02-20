from torchvision import datasets, transforms
from torchvision.models import resnet18
from src.unlearning.utils import remove_samples_by_indices, remove_classes
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(224),  # Resize CIFAR images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

# Load CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_dataset = Subset(train_dataset, list(range(2000)))  # Reduce the size of the train dataset
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_dataset = Subset(test_dataset, list(range(1000)))  # Reduce the size of the test dataset

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 3

for epoch in range(num_epochs):
    running_loss = 0.
    correct = 0
    total = 0
    model.train()
    print(f'Epoch {epoch + 1}')
    for images, labels in train_loader:
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Statistics
        running_loss += loss.item()
        _, predictions = outputs.max(1)
        total += labels.size(0)  # This returns the batch size
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss/len(train_loader)
    epoch_accuracy = 100. * correct/total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# MODEL TESTING
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = outputs.max(1)
        total += labels.size(0)
        correct += (labels == predictions).sum().item()

    test_acc = 100. * correct/total

print(f"Test Accuracy on CIFAR-10: {test_acc:.2f}%")

# torch.save(model.state_dict(), "resnet18_ft_cifar10.pth")
# print('Model saved!')