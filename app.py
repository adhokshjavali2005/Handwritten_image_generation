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

model = None   # <-- lazy-loaded

# Text config
MAX_LEN = 50
characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'-"
vocab = sorted(list(set(characters)))
char_to_idx = {c: i + 1 for i, c in enumerate(vocab)}

def encode_text(text):
    text = text[:MAX_LEN]
    seq = [char_to_idx.get(c, 0) for c in text]
    return seq + [0] * (MAX_LEN - len(seq))

def load_model():
    global model
    if model is None:
        print("🔄 Loading model...")
        model = tf.keras.models.load_model("text_to_handwriting.keras")
        print("✅ Model loaded")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Handwritten Text Generator API is running"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/generate", methods=["POST"])
def generate():
    load_model()  # <-- model loads only when needed

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    seq = np.array([encode_text(text)])
    pred = model.predict(seq)[0]

    img = (pred.squeeze() * 255).astype("uint8")
    image = Image.fromarray(img)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": encoded})
