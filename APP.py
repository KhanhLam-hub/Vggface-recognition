from flask import Flask, request, jsonify
from deepface import DeepFace
import numpy as np
import pickle
import cv2
import os
import requests
from io import BytesIO

# ================== CẤU HÌNH ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")       # Token bot Telegram
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")   # Chat ID nhận cảnh báo
API_KEY_UPLOAD = os.getenv("API_KEY_UPLOAD")       # API key xác thực từ ESP32
GITHUB_EMBEDDINGS_URL = os.getenv("GITHUB_EMBEDDINGS_URL")  # Link RAW tới embeddings.pkl trên GitHub

app = Flask(__name__)

# ================== TẢI EMBEDDINGS TỪ GITHUB ==================
def load_embeddings_from_github():
    """Tải embeddings.pkl từ GitHub"""
    try:
        res = requests.get(GITHUB_EMBEDDINGS_URL)
        res.raise_for_status()
        embeddings_data = pickle.loads(res.content)
        print("✅ Tải embeddings từ GitHub thành công")
        return embeddings_data
    except Exception as e:
        print("❌ Lỗi tải embeddings từ GitHub:", e)
        return None

# ================== GỬI ẢNH + CẢNH BÁO TELEGRAM ==================
def send_telegram_alert(message, image=None):
    """Gửi tin nhắn + ảnh cảnh báo tới Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

        if image is not None:
            files = {"photo": ("alert.jpg", image, "image/jpeg")}
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files=files)
    except Exception as e:
        print("❌ Lỗi gửi Telegram:", e)

# ================== NẠP MÔ HÌNH BAN ĐẦU ==================
embeddings_data = load_embeddings_from_github()
if embeddings_data:
    person_names = embeddings_data["person_names"]
    stored_embeddings = np.array(embeddings_data["embeddings"])
else:
    person_names = []
    stored_embeddings = np.array([])

# ================== API NHẬN ẢNH ==================
@app.route("/upload", methods=["POST"])
def upload_image():
    """Nhận ảnh từ ESP32-CAM, so khớp khuôn mặt"""
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY_UPLOAD}":
        return "❌ Unauthorized", 401

    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh gửi lên"}), 400

    file = request.files["image"].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        detections = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)
        if not detections:
            return jsonify({"result": "Không phát hiện khuôn mặt"})

        face_embedding = np.array(detections[0]["embedding"])
        distances = np.linalg.norm(stored_embeddings - face_embedding, axis=1)

        if len(distances) == 0:
            send_telegram_alert("🚨 Không có dữ liệu khuôn mặt!", file)
            return jsonify({"result": "Không có dữ liệu khuôn mặt"})

        min_dist = np.min(distances)
        idx = np.argmin(distances)
        name = person_names[idx] if min_dist < 0.5 else "Người lạ"

        if name == "Người lạ":
            send_telegram_alert("🚨 Phát hiện NGƯỜI LẠ!", file)
        else:
            send_telegram_alert(f"✅ Nhận diện: {name}", file)

        return jsonify({"name": name, "distance": float(min_dist)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================== CHẠY SERVER ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

