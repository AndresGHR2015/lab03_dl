import os
import pickle
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models import MultiTaskResNet


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/resnet18_age_gender.pth"))
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

    try:
        predicted_gender, estimated_age = predict(uploaded_image)
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
