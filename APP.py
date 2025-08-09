from flask import Flask, request, jsonify
import numpy as np
import pickle
import cv2
import os
import requests

# ================== CẤU HÌNH ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")      
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  
API_KEY_UPLOAD = os.getenv("API_KEY_UPLOAD")       

# Link raw file embeddings trên GitHub (thay bằng link của bạn)
EMBEDDINGS_URL = "https://raw.githubusercontent.com/username/repo/main/embeddings.pkl"

app = Flask(__name__)

# ================== TẢI EMBEDDINGS TỪ GITHUB VÀ LOAD TRỰC TIẾP ==================
def load_embeddings_from_github():
    try:
        print("⏳ Đang tải embeddings trực tiếp từ GitHub...")
        r = requests.get(EMBEDDINGS_URL, timeout=10)
        r.raise_for_status()
        embeddings_data = pickle.loads(r.content)
        print(f"✅ Tải embeddings thành công ({len(embeddings_data['person_names'])} entries).")
        return embeddings_data
    except Exception as e:
        print("❌ Lỗi tải embeddings từ GitHub:", e)
        return None

# ================== GỬI ẢNH + CẢNH BÁO TELEGRAM ==================
def send_telegram_alert(message, image=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        if image is not None:
            files = {"photo": ("alert.jpg", image, "image/jpeg")}
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files=files, timeout=5)
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
    print("⚠️ Không có dữ liệu embeddings, ứng dụng sẽ chạy nhưng không nhận diện được khuôn mặt")

# ================== ROUTE CƠ BẢN CHO / ==================
@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "Server is running", "embeddings_loaded": bool(embeddings_data)})

# ================== API NHẬN ẢNH ==================
@app.route("/upload", methods=["POST"])
def upload_image():
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY_UPLOAD}":
        return jsonify({"error": "Unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh gửi lên"}), 400

    file = request.files["image"].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không thể giải mã ảnh"}), 400

    try:
        from deepface import DeepFace
        detections = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)
        if not detections:
            send_telegram_alert("🚨 Không phát hiện khuôn mặt!", file)
            return jsonify({"result": "Không phát hiện khuôn mặt"})

        face_embedding = np.array(detections[0]["embedding"])
        if len(stored_embeddings) == 0:
            send_telegram_alert("🚨 Không có dữ liệu khuôn mặt!", file)
            return jsonify({"result": "Không có dữ liệu khuôn mặt"})

        distances = np.linalg.norm(stored_embeddings - face_embedding, axis=1)
        min_dist = np.min(distances)
        idx = np.argmin(distances)
        name = person_names[idx] if min_dist < 0.5 else "Người lạ"

        if name == "Người lạ":
            send_telegram_alert("🚨 Phát hiện NGƯỜI LẠ!", file)
        else:
            send_telegram_alert(f"✅ Nhận diện: {name}", file)

        return jsonify({"name": name, "distance": float(min_dist)})
    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return jsonify({"error": str(e)}), 500

# ================== CHẠY SERVER ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
