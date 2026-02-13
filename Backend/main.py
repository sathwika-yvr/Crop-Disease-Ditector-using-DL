import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
import shutil
import os
from typing import Dict

# ------------------------
# Load model and mappings
# ------------------------
MODEL_PATH = "crop_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
REMEDIES_PATH = "remedies.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
with open(REMEDIES_PATH) as f:
    remedies = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ------------------------
# FastAPI app setup
# ------------------------
app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Simple in-memory DB
# ------------------------
farmers_db: Dict[str, Dict] = {}  # { phone: {"history": [...] } }

# ------------------------
# Models
# ------------------------
class LoginRequest(BaseModel):
    phone: str

class PredictResult(BaseModel):
    disease: str
    probability: float
    remedy: str

# ------------------------
# Routes
# ------------------------
@app.post("/login")
def login(req: LoginRequest):
    phone = req.phone
    if phone not in farmers_db:
        farmers_db[phone] = {"history": []}
    return {"message": f"Farmer {phone} logged in successfully."}


@app.post("/predict/{phone}")
async def predict(phone: str, file: UploadFile = File(...)):
    # Save uploaded image temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probability = float(np.max(predictions[0]))
    disease = idx_to_class[predicted_class]
    remedy = remedies.get(disease, "No remedy available.")

    # Save to farmer history
    if phone in farmers_db:
        farmers_db[phone]["history"].append({
            "disease": disease,
            "confidence": probability,
            "remedy": remedy
        })

    # Cleanup
    os.remove(file_path)

    return {
        "disease": disease,
        "probability": probability,
        "remedy": remedy
    }


@app.get("/history/{phone}")
def get_history(phone: str):
    if phone in farmers_db:
        return {"history": farmers_db[phone]["history"]}
    return {"history": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
