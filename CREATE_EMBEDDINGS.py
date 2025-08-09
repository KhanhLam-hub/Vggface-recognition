# Cài thư viện cần thiết
import os
import pickle
import numpy as np
from deepface import DeepFace
import cv2

def create_embeddings(image_dir, output_path, detector_backend="opencv"):
    """Tạo và lưu embeddings từ thư mục ảnh"""
    person_names = []
    embeddings = []
    labels = []

    label_id = 0
    for person_name in os.listdir(image_dir):
        person_path = os.path.join(image_dir, person_name)
        if not os.path.isdir(person_path):
            print(f"⚠️ Bỏ qua {person_path}: không phải thư mục")
            continue

        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            # Kiểm tra file ảnh có hợp lệ không
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Ảnh {img_path} không đọc được (hỏng hoặc định dạng sai)")
                continue
            try:
                # DeepFace tự detect mặt và tạo embedding
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="VGG-Face",
                    detector_backend=detector_backend, 
                    enforce_detection=False  
                )[0]["embedding"]
                embeddings.append(embedding)
                person_names.append(person_name)
                labels.append(label_id)
                print(f"✅ Xử lý thành công {img_path}")
            except Exception as e:
                print(f"❌ Lỗi tạo embedding cho {img_path}: {e}")
        label_id += 1

    if not embeddings:
        print("❌ Không tạo được embedding nào. Kiểm tra lại thư mục ảnh.")
        return
    # Tạo dictionary
    data = {
        "person_names": person_names,
        "embeddings": np.array(embeddings),
        "labels": labels
    }

    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Lưu embeddings thành công tại {output_path}")
    except Exception as e:
        print(f"❌ Lỗi lưu embeddings: {e}")

if __name__ == "__main__":
    # Đường dẫn tới thư mục chứa thư mục ảnh người thân
    image_dir = "/content/drive/MyDrive/Nhandienkhuonmat/Data_faces"
    # File pickle để lưu embeddings
    output_path = "/content/drive/MyDrive/Nhandienkhuonmat/embeddings.pkl"
    create_embeddings(image_dir, output_path, detector_backend="opencv")
