import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI(title="Crop Disease Detection API")

# Use absolute path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "torch_model.pth")
CLASS_NAMES = [
    'Apple Black_spot', 'Apple Brown_spot', 'Apple Normal',
    'Apricot blight leaf disease', 'Apricot Normal', 'Apricot shot_hole',
    'Bean bean rust image', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean shot_hole',
    'Cherry brown_spot', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry purple leaf spot', 'Cherry_shot hole disease',
    'Corn Fungal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Corn Normal leaf',
    'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf',
    'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease',
    'lokat Leaf_spot', 'Lokat Normal leaf',
    'Pear Black spot _ leaf disease', 'Pear fire blight', 'Pear Normal _leaf',
    'persimmons Brown_spot',
    'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf',
    'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut leaf gall mite', 'Walnut Normal_leaf', 'Walnut Shot_hole'
]

model = None
device = None

# Image preprocessing (must match training transforms)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_model():
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            # Create MobileNetV2 model with same architecture as training
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

            # Load trained weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()

            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model not found at {MODEL_PATH}")

@app.get("/")
async def root():
    return {"message": "Crop Disease Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        img_tensor = preprocess(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        confidence, predicted_idx = torch.max(probabilities, 0)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_pct = confidence.item() * 100

        # Split class name into crop and disease
        parts = predicted_class.split(' ', 1)
        if len(parts) == 2:
            crop, disease = parts
        else:
            parts = predicted_class.split('_', 1)
            crop, disease = parts if len(parts) == 2 else (predicted_class, "Unknown")

        return {
            "success": True,
            "message": "Crop analyzed successfully",
            "data": {
                "crop": crop,
                "disease": disease,
                "confidence": f"{confidence_pct:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
