import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI(title="Crop Disease Detection API")

MODEL_PATH = "models/saved_model"
CLASS_NAMES = [
    'Apple Brown_spot', 'Apple Normal', 'Apple black_spot', 
    'Apricot Normal', 'Apricot blight leaf disease', 'Apricot shot_hole', 
    'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean bean rust image', 'Bean shot_hole', 
    'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry brown_spot', 'Cherry purple leaf spot', 'Cherry_shot hole disease', 
    'Corn Fungal leaf', 'Corn Normal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 
    'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf', 
    'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease', 
    'Lokat Normal leaf', 
    'Pear Black spot _ leaf disease', 'Pear Normal _leaf', 'Pear fire blight', 
    'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut Normal_leaf', 'Walnut Shot_hole', 'Walnut leaf gall mite', 
    'lokat Leaf_spot', 
    'persimmons Brown_spot', 
    'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf'
]

model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            # We don't raise here to allow API to start, but predict will fail
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
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        # Split class name into crop and disease
        # Class names follow pattern: "Crop Disease" or "Crop_Disease"
        parts = predicted_class.split(' ', 1)
        if len(parts) == 2:
            crop, disease = parts
        else:
            # Handle cases with underscores like "tomato_healthy_leaf"
            parts = predicted_class.split('_', 1)
            crop, disease = parts if len(parts) == 2 else (predicted_class, "Unknown")

        return {
            "success": True,
            "message": "Crop analyzed successfully",
            "data": {
                "crop": crop,
                "disease": disease,
                "confidence": f"{confidence:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
