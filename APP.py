from flask import Flask, request, jsonify
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
LOCAL_EMBEDDINGS_PATH = "/opt/render/.deepface/weights/embeddings.pkl"  # Đường dẫn cục bộ dự phòng

app = Flask(__name__)

# ================== TẢI EMBEDDINGS TỪ GITHUB HOẶC CỤC BỘ ==================
def load_embeddings_from_github():
    """Tải embeddings.pkl từ GitHub hoặc file cục bộ nếu thất bại"""
    try:
        res = requests.get(GITHUB_EMBEDDINGS_URL, timeout=10)
        res.raise_for_status()
        
        # Kiểm tra nội dung có phải nhị phân hợp lệ không
        if res.headers.get("content-type", "").startswith("text") or b"<html" in res.content[:100]:
            raise ValueError("Tải về nội dung không phải file pickle (có thể là HTML)")
        
        embeddings_data = pickle.loads(res.content)
        print("✅ Tải embeddings từ GitHub thành công")
        return embeddings_data
    except Exception as e:
        print("❌ Lỗi tải embeddings từ GitHub:", e)
        # Thử tải từ file cục bộ
        return load_embeddings_from_local()

def load_embeddings_from_local():
    """Tải embeddings từ file cục bộ nếu có"""
    try:
        if os.path.exists(LOCAL_EMBEDDINGS_PATH):
            with open(LOCAL_EMBEDDINGS_PATH, "rb") as f:
                embeddings_data = pickle.load(f)
            print("✅ Tải embeddings từ file cục bộ thành công")
            return embeddings_data
        else:
            print("❌ Không tìm thấy file embeddings cục bộ tại", LOCAL_EMBEDDINGS_PATH)
            return None
    except Exception as e:
        print("❌ Lỗi tải embeddings cục bộ:", e)
        return None

# ================== GỬI ẢNH + CẢNH BÁO TELEGRAM ==================
def send_telegram_alert(message, image=None):
    """Gửi tin nhắn + ảnh cảnh báo tới Telegram"""
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
    """Route cơ bản để kiểm tra server"""
    return jsonify({"status": "Server is running", "embeddings_loaded": bool(embeddings_data)})

# ================== API NHẬN ẢNH ==================
@app.route("/upload", methods=["POST"])
def upload_image():
    """Nhận ảnh từ ESP32-CAM, so khớp khuôn mặt"""
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
        from deepface import DeepFace  # Import chậm để giảm thời gian khởi động
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
