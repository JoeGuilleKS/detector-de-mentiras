from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Cargar modelo entrenado
model = load_model('model/Lie_Truth.keras')

# Preprocesamiento de imagen
def preprocess_image(img):
    img = img.resize((224, 224))  # adapta el tamaño según tu modelo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalización si tu modelo lo necesita
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
