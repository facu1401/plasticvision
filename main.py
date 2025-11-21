from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Carpeta para guardar las imágenes subidas
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta del modelo
MODEL_PATH = 'keras_model.h5'

# Verificar si el modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

# Cargar el modelo
model = load_model(MODEL_PATH)

# Cargar etiquetas desde labels.txt
LABELS_PATH = 'labels.txt'
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def predict_image(img_path):
    """Predice la clase y confianza de la imagen"""
    img = image.load_img(img_path, target_size=(224, 224))  # Ajustar tamaño según tu modelo
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalización si tu modelo la requiere

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]

    return CLASS_NAMES[class_idx], confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    label = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Guardar archivo de forma segura
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predecir con el modelo
            label, confidence = predict_image(file_path)
            confidence = round(confidence * 100, 2)  # Convertir a porcentaje

    return render_template('index.html',
                           filename=filename,
                           label=label,
                           confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
