import os
import pickle
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.models import MultiTaskResNet
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extraer_rostro(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)
    
    if len(rostros) == 0:
        return None

    x, y, w, h = rostros[0]

    margen = 20
    y1 = max(0, y - margen)
    y2 = min(img_cv.shape[0], y + h + margen)
    x1 = max(0, x - margen)
    x2 = min(img_cv.shape[1], x + w + margen)
    
    cara_recortada = img_cv[y1:y2, x1:x2]

    return Image.fromarray(cv2.cvtColor(cara_recortada, cv2.COLOR_BGR2RGB))


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "outputs/models/ResNet_FineTuned_best.pth"))
GENDER_LABELS = {
    0: "Hombre",
    1: "Mujer",
}


def preprocess_image(image):
    validation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return validation_transform(image.convert("RGB")).unsqueeze(0)


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en {MODEL_PATH}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskResNet(freeze_backbone=False, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(get_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model, device


def predict(image):
    model, device = load_model()
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        gender_output, age_output = model(image_tensor)

    gender_index = int(torch.argmax(gender_output, dim=1).item())
    gender = GENDER_LABELS.get(gender_index, "Desconocido")
    age = float(age_output.reshape(-1).item())
    return gender, age


st.title("Inferencia de género y edad")

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
    except (OSError, ValueError):
        st.error("No se pudo abrir la imagen seleccionada.")
        st.stop()

    st.image(uploaded_image, caption="Imagen cargada", use_container_width=True)

    rostro = extraer_rostro(uploaded_image)
    if rostro is not None:
        imagen_para_prediccion = rostro
        st.image(rostro, caption="Rostro detectado", use_container_width=True)
    else:
        imagen_para_prediccion = uploaded_image
        st.warning("No se detectó un rostro. Se usará la imagen completa.")

    try:
        predicted_gender, estimated_age = predict(imagen_para_prediccion)
        st.write(f"Género predicho: {predicted_gender}")
        st.write(f"Edad estimada: {estimated_age:.1f}")
    except FileNotFoundError as error:
        st.error(str(error))
    except (
        EOFError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ):
        st.error("No se pudo cargar el modelo entrenado para realizar la inferencia.")
