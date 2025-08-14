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

EMBEDDINGS_URL = "https://raw.githubusercontent.com/KhanhLam-hub/Vggface-recognition/main/embeddings.pkl"

app = Flask(__name__)

# ================== GỬI CẢNH BÁO TELEGRAM ==================
def send_telegram_alert(message, image_bytes=None):
    try:
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            # Gửi text
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                timeout=5
            )
            # Gửi ảnh nếu có
            if image_bytes:
                files = {"photo": ("alert.jpg", image_bytes, "image/jpeg")}
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files=files,
                    timeout=5
                )
    except Exception as e:
        print("❌ Lỗi gửi Telegram:", e)

# ================== THÔNG BÁO KHỞI ĐỘNG ==================
def send_startup_message():
    send_telegram_alert("🌐 Server nhận diện khuôn mặt đã khởi động thành công và sẵn sàng hoạt động!")

send_startup_message()

# ================== LOAD EMBEDDINGS ==================
def load_embeddings():
    try:
        print("⏳ Đang tải embeddings từ GitHub...")
        r = requests.get(EMBEDDINGS_URL, timeout=10)
        r.raise_for_status()
        data = pickle.loads(r.content)
        print(f"✅ Tải thành công ({len(data['person_names'])} entries).")
        return data
    except Exception as e:
        print("❌ Lỗi tải embeddings:", e)
        return None

embeddings_data = load_embeddings()
if embeddings_data:
    person_names = embeddings_data["person_names"]
    stored_embeddings = np.array(embeddings_data["embeddings"])
else:
    person_names = []
    stored_embeddings = np.array([])
    print("⚠️ Không có dữ liệu embeddings, server vẫn chạy nhưng không nhận diện được khuôn mặt")

# ================== ROUTE KIỂM TRA ==================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server running 🚀", "embeddings_loaded": bool(embeddings_data)})

# ================== ROUTE NHẬN ẢNH ==================
@app.route("/upload", methods=["POST"])
def upload_image():
    # Kiểm tra API key
    auth = request.headers.get("Authorization")
    if auth != f"Bearer {API_KEY_UPLOAD}":
        return jsonify({"error": "Unauthorized"}), 401
    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh gửi lên"}), 400

    # Đọc ảnh
    file = request.files["image"].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không thể giải mã ảnh"}), 400

    # Resize ảnh để giảm RAM
    try:
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print("❌ Lỗi resize ảnh:", e)
        return jsonify({"error": "Không thể resize ảnh"}), 500

    # Encode ảnh để gửi Telegram
    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()

    # Xử lý nhận diện
    try:
        from deepface import DeepFace
        detections = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)
        if not detections:
            send_telegram_alert("🚨 Không phát hiện khuôn mặt!", img_bytes)
            return jsonify({"result": "Không phát hiện khuôn mặt"})

        face_embedding = np.array(detections[0]["embedding"])
        if len(stored_embeddings) == 0:
            send_telegram_alert("🚨 Không có dữ liệu khuôn mặt!", img_bytes)
            return jsonify({"result": "Không có dữ liệu khuôn mặt"})

        # So sánh khoảng cách
        distances = np.linalg.norm(stored_embeddings - face_embedding, axis=1)
        min_dist = np.min(distances)
        idx = np.argmin(distances)
        name = person_names[idx] if min_dist < 0.5 else "Người lạ"

        # Gửi cảnh báo Telegram
        if name == "Người lạ":
            send_telegram_alert("🚨 Phát hiện NGƯỜI LẠ!", img_bytes)
        else:
            send_telegram_alert(f"✅ Nhận diện: {name}", img_bytes)

        return jsonify({"name": name, "distance": float(min_dist)})
    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return jsonify({"error": str(e)}), 500

# ================== CHẠY SERVER ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
