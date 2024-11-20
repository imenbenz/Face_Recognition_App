import streamlit as st
import cv2
from keras_facenet import FaceNet
import joblib
import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Charger le modèle et le label encoder une seule fois
@st.cache_resource
def load_model():
    model = joblib.load('models/svm_face_recognition_model.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    embedder = FaceNet()
    detector = MTCNN()
    return model, encoder, embedder, detector

model, encoder, embedder, detector = load_model()

# Fonction pour obtenir les embeddings
def get_embedding(face_img):
    face_img = face_img.astype('float32')  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (None x 160 x 160 x 3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D vector

# Détection de visage et reconnaissance
def recognize_faces(image, threshold=50):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return "No faces detected", image

    faces_info = []  # Liste pour stocker les informations sur chaque visage détecté
    for result in results:
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        embedding = get_embedding(face)

        probabilities = model.predict_proba([embedding])[0]
        confidence = max(probabilities) * 100
        predicted_class = model.predict([embedding])[0]
        name = encoder.inverse_transform([predicted_class])[0]

        if confidence < threshold:  # Si la confiance est inférieure au seuil
            name = "Unknown"

        # Dessiner le rectangle et le label sur l'image
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{name} ({confidence:.2f}%)"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        faces_info.append((name, confidence))  # Ajouter l'info du visage à la liste

    return faces_info, image

# Interface Streamlit avec design amélioré
st.set_page_config(page_title="Face Recognition App", page_icon=":guardsman:", layout="wide")

st.title("Face Recognition App")
st.markdown("""
    Bienvenue dans l'application de reconnaissance faciale !  
    Vous pouvez télécharger une image contenant un ou plusieurs visages, et l'application les identifiera avec une certaine précision.
""")

# Choisir un seuil de confiance avec le curseur
threshold = st.slider("Seuil de confiance", min_value=0, max_value=100, value=50, step=1)

# Télécharger l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Charger l'image téléchargée
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Afficher l'image téléchargée
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Affichage de la barre de progression pendant le traitement
    with st.spinner("Traitement de l'image..."):
        # Reconnaissance faciale (plusieurs visages)
        faces_info, annotated_image = recognize_faces(image, threshold)

    # Afficher l'image annotée
    st.image(annotated_image, caption="Résultat de la reconnaissance", use_column_width=True)

    # Afficher les résultats de la reconnaissance
    if len(faces_info) > 0:
        st.subheader("Visages détectés:")
        for idx, (name, confidence) in enumerate(faces_info):
            st.write(f"Visage {idx + 1}: {name} (Confiance: {confidence:.2f}%)")
    else:
        st.write("Aucun visage détecté.")

# Ajouter une section à propos
st.sidebar.header("À propos de l'application")
st.sidebar.write("""
    Cette application utilise des modèles de reconnaissance faciale pour détecter et reconnaître les visages dans les images téléchargées.
    Elle est alimentée par le modèle FaceNet et utilise un classificateur SVM pour la reconnaissance des visages.
    Le seuil de confiance peut être ajusté pour changer la sensibilité de la reconnaissance.
""")
