import torch
import timm
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=3
)

model.load_state_dict(
    torch.load("crop_disease_model.pth", map_location=device)
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open("test.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    pred = torch.argmax(outputs, dim=1)

print("Disease:", classes[pred.item()])