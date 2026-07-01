import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder(
    root='dataset/train',
    transform=transform
)

valid_dataset = datasets.ImageFolder(
    root='dataset/valid',
    transform=transform
)

# Data loader
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
print(len(train_loader))
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

# Load Vision Transformer model
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=3
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 3

for epoch in range(epochs):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
    

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        print(f"Loss: {loss.item():.4f}")

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'models/crop_disease_model.pth')

print("Model Training Complete")
print("Model Saved Successfully")