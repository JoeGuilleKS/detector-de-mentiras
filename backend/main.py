from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Permitir CORS desde el frontend de Azure
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner el dominio exacto de Azure si quieres más seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("backend/model/Lie_Truth.keras")  # Ruta relativa correcta

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224))  # Ajusta tamaño según tu modelo
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = "Mentira" if prediction[0][0] > 0.5 else "Verdad"

    return {"prediction": label}
