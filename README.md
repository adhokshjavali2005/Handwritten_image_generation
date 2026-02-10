✍️ Handwritten Image Generation using Deep Learning

This project converts typed text into handwritten-style images using a deep learning model.
It demonstrates an end-to-end AI system with a deployed backend and a modern frontend.

🚀 Live Demo

Frontend (Lovable): [Handwritten AI Studio](https://handwritten-image-generation.lovable.app)

Backend (Render):
https://handwritten-image-generation.onrender.com

⚠️ Note: The backend runs on Render Free tier, so the first request may take 20–50 seconds due to cold start.

🧠 Project Overview

The system takes user input text and generates a handwritten image that resembles natural handwriting.

Key Highlights

Character-level text encoding

CNN-based handwriting image generation

Flask REST API for inference

Deployed backend using Render

Frontend built using Lovable (no-code UI builder)

🏗️ Architecture
User Text
   ↓
Text Encoder (Character-level encoding)
   ↓
Deep Learning Model (CNN-based)
   ↓
Generated Handwritten Image
   ↓
Base64 PNG Response
   ↓
Frontend Preview

🧪 Technologies Used
🔹 Backend

Python

TensorFlow / Keras

Flask

Flask-CORS

Pillow (PIL)

NumPy

Gunicorn

🔹 Frontend

Lovable (No-code frontend builder)

Fetch-based API integration

🔹 Deployment

Render (Backend API hosting)

GitHub (Version control)

📦 Model Details

Input: Encoded text sequence (max length = 50 characters)

Output: Handwritten-style grayscale image

Model type: CNN-based generative model

Output format: Base64-encoded PNG image

To improve realism, controlled random noise is added during inference so the same text does not always produce the exact same image.
