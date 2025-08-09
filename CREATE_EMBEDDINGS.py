
# Cài thư viện cần thiết
!pip install mtcnn keras-facenet

import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from keras_facenet import FaceNet
import os

# Hàm cắt khuôn mặt từ ảnh
def extract_face(filename, required_size=(160, 160)):
    pixels = Image.open(filename).convert('RGB')
    pixels = asarray(pixels)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        print(f"❌ Không tìm thấy khuôn mặt trong {filename}")
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return asarray(image)

# Load model FaceNet
embedder = FaceNet()

# Thư mục chứa ảnh người thân
image_dir = "/content/drive/MyDrive/Nhandienkhuonmat/Data_faces" 

# Lưu embedding
person_names = []
embeddings = []
labels = []

label_id = 0
for person_name in os.listdir(image_dir):
    person_path = os.path.join(image_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        face_pixels = extract_face(img_path)
        if face_pixels is not None:
            embedding = embedder.embeddings([face_pixels])[0]
            embeddings.append(embedding)
            person_names.append(person_name)
            labels.append(label_id)
    label_id += 1

# Tạo dictionary
data = {
    "person_names": person_names,
    "embeddings": np.array(embeddings),
    "labels": labels
}
