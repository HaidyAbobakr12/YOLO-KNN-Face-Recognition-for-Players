import streamlit as st
import cv2
import numpy as np
import joblib
import insightface
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches


knn = joblib.load("face_recognition_knn.pkl")
encoder = joblib.load("label_encoder.pkl")
detector = YOLO("best.pt")

# InsightFace
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640,640))

st.title("Face Recognition System")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # حفظ الصورة مؤقتاً
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = cv2.imread("temp.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector(img_rgb)
    boxes_list = []
    names_list = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_rgb.shape[1], x2 + pad)
            y2 = min(img_rgb.shape[0], y2 + pad)

            face_crop = img_rgb[y1:y2, x1:x2]
            faces = model.get(face_crop)
            if len(faces) == 0:
                continue
            emb = faces[0].embedding
            pred = knn.predict([emb])[0]
            name = encoder.inverse_transform([pred])[0]

            boxes_list.append([x1, y1, x2, y2])
            names_list.append(name)

    # عرض النتيجة
    fig, ax = plt.subplots(1, figsize=(12,12))
    ax.imshow(img_rgb)
    ax.axis("off")

    for box, name in zip(boxes_list, names_list):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-10, name, color="yellow", fontsize=12,
                bbox=dict(facecolor="black", alpha=0.5, pad=1))

    st.pyplot(fig)

    # كمان نعرض أسماء الناس تحت الصورة
    st.subheader(" Recognized Faces")
    for name in names_list:
        st.write(f"- {name}")
