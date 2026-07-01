from flask import Flask, render_template, request
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]

disease_info = {
    "Tomato_Early_Blight": {
        "cause": "Fungal Infection",
        "treatment": "Use fungicide and remove infected leaves.",
        "prevention": "Avoid overhead watering and keep leaves dry."
    },

    "Tomato_Late_Blight": {
        "cause": "Water Mold Disease",
        "treatment": "Apply copper-based fungicide.",
        "prevention": "Maintain proper air circulation."
    },

    "Tomato_Healthy": {
        "cause": "Healthy Plant",
        "treatment": "No treatment required.",
        "prevention": "Continue proper irrigation and nutrition."
    }
}

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

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)

            probs = torch.softmax(output, dim=1)

            confidence = torch.max(probs).item() * 100
            if confidence >= 85:
                severity = "Low Risk 🟢"
            elif confidence >= 70:
                severity = "Medium Risk 🟡"
            else:
                severity = "High Risk 🔴"

            pred = torch.argmax(output, dim=1)

        disease_name = classes[pred.item()]
        display_name = disease_name.replace("_"," ")

        prediction = {
            "disease": display_name,
            "confidence": round(confidence, 2),
            "image": filepath,
            "cause": disease_info[disease_name]["cause"],
            "treatment": disease_info[disease_name]["treatment"],
            "prevention": disease_info[disease_name]["prevention"],
            "severity":severity,
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)