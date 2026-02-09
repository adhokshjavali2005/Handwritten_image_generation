from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import string
import os
import cv2

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("text_to_handwriting.keras")

# -----------------------------
# Text config
# -----------------------------
MAX_LEN = 50
characters = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + " .,!?'-"
)
vocab = sorted(list(set(characters)))
char_to_idx = {c: i + 1 for i, c in enumerate(vocab)}

def encode_text(text):
    text = text[:MAX_LEN]
    seq = [char_to_idx.get(c, 0) for c in text]
    return seq + [0] * (MAX_LEN - len(seq))

# -----------------------------
# Image enhancement (NEW)
# -----------------------------
def enhance_handwriting(img):
    """
    img: float image in range [0,1], shape (H, W)
    returns: uint8 enhanced image
    """
    img = np.clip(img, 0, 1)

    # Convert to uint8
    img_u8 = (img * 255).astype(np.uint8)

    # Contrast normalization
    img_norm = cv2.normalize(
        img_u8, None, 0, 255, cv2.NORM_MINMAX
    )

    # Unsharp masking for stroke clarity
    blurred = cv2.GaussianBlur(img_norm, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        img_norm, 1.6,
        blurred, -0.6,
        0
    )

    return sharpened

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Handwritten Text Generator API is running"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/generate", methods=["POST", "GET"])
def generate():
    if request.method == "GET":
        return jsonify({
            "message": "Use POST with JSON: { 'text': 'Hello world' }"
        })

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    print("Received text:", text)

    # Encode text
    seq = np.array([encode_text(text)])

    # Predict handwriting
    pred = model.predict(seq)[0]

    # 🔥 Enhance visibility
    enhanced_img = enhance_handwriting(pred.squeeze())

    # Convert to PNG
    image = Image.fromarray(enhanced_img)
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({"image": encoded})

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
