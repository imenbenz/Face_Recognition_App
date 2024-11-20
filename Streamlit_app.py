import streamlit as st
import cv2
from keras_facenet import FaceNet
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Charger le modèle et le label encoder
model = joblib.load('models/svm_face_recognition_model.pkl')
encoder = joblib.load('models/label_encoder.pkl')
embedder = FaceNet()
detector = MTCNN()

# Fonction pour obtenir les embeddings
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (None x 160 x 160 x 3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D vector

# Détection de visage et reconnaissance
def recognize_face(image, threshold=50):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return "No face detected", image

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    embedding = get_embedding(face)

    probabilities = model.predict_proba([embedding])[0]
    confidence = max(probabilities) * 100
    predicted_class = model.predict([embedding])[0]
    name = encoder.inverse_transform([predicted_class])[0]

    if confidence < threshold:  # If confidence is below the threshold
        name = "Unknown"

    # Draw rectangle and label on the image
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{name} ({confidence:.2f}%)"
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return name, image

# Interface Streamlit
st.title("Face Recognition App")
st.write("Upload an image to recognize faces.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Reconnaissance faciale
    name, annotated_image = recognize_face(image)

    # Afficher les résultats
    st.image(annotated_image, caption=f"Result: {name}", use_column_width=True)

# Interface Streamlit
st.title("Face Recognition App")
st.write("Capture a photo using your camera for real-time face recognition!")

# Capture vidéo en temps réel avec la caméra
camera_input = st.camera_input("Take a picture")

if camera_input is not None:
    # Convertir l'image capturée en un tableau NumPy
    image = Image.open(camera_input)
    image = np.array(image)

    # Reconnaissance faciale
    name, annotated_image = recognize_face(image)

    # Afficher l'image annotée avec les résultats
    st.image(annotated_image, caption=f"Result: {name}", use_column_width=True)
