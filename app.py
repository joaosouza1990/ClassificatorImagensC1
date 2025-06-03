from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Carregar o modelo treinado
model = tf.keras.models.load_model("seu_modelo/modelo_salvo")  # Substitua pelo caminho do modelo

def preprocess_image(image):
    image = image.resize((224, 224))  # Ajuste para o tamanho do modelo
    image = np.array(image) / 255.0  # Normalização
    return np.expand_dims(image, axis=0)  # Adicionar dimensão extra

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        classes = ["Pessoa", "Não Pessoa"]

        return render_template("index.html", result=classes[class_index], confidence=float(prediction[0][class_index]))

    return render_template("index.html", result=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
