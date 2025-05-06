from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# CORS: permite que tu frontend en Azure pueda conectarse a este backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Puedes restringir a tu dominio de Azure
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo una vez al iniciar el servidor
model = load_model("model/Lie_Truth.keras")

# Procesar imagen como lo hiciste en Colab
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_imagefile(await file.read())
    img = img.resize((224, 224))  # Ajusta esto a lo que usaste para entrenar
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]  # Suponiendo salida binaria
    result = "Verdad" if prediction < 0.5 else "Mentira"

    return {"resultado": result}
