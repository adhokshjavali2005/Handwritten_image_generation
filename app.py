from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import string
import os

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model("text_to_handwriting.keras")

# Text config
MAX_LEN = 50
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'-"
vocab = sorted(list(set(characters)))
char_to_idx = {c: i+1 for i, c in enumerate(vocab)}

def encode_text(text):
    text = text[:MAX_LEN]
    seq = [char_to_idx.get(c, 0) for c in text]
    return seq + [0] * (MAX_LEN - len(seq))

@app.route("/")
def home():
    return "Handwritten Text Generator API is running"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    text = data.get("text", "")

    seq = np.array([encode_text(text)])
    pred = model.predict(seq)[0]

    img = (pred.squeeze() * 255).astype("uint8")
    image = Image.fromarray(img)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": encoded})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
