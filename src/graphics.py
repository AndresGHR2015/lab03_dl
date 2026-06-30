import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PLOTS_OUTPUT_DIR = Path("outputs/plots_finales")
METRICS_DIR = Path("outputs/models")

MODEL_KEYS = ["mlp", "cnn", "resnet", "ResNet_FineTuned", "CNN_Ablacion"]
MODEL_DISPLAY_NAMES = [
    "E2: MLP Multitarea",
    "E3: CNN Simple",
    "E4: ResNet (Congelada)",
    "E5: ResNet (Fine-Tuning)",
    "E6: Ablacion (lambda=0.5)",
]

BASELINE_DISPLAY_NAME = "E1: Baseline (PCA + GNB/Ridge)"
BASELINE_GENDER_ACCURACY = 0.8128
BASELINE_AGE_MAE = 11.7690

RESNET_FINETUNED_KEY = "ResNet_FineTuned"


def _load_json(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def _load_model_metrics():
    gender_accuracies = []
    age_maes = []

    for model_key in MODEL_KEYS:
        metrics_path = METRICS_DIR / f"{model_key}_metrics.json"
        try:
            data = _load_json(metrics_path)
            gender_accuracies.append(data.get("gender_accuracy", 0.0))
            age_maes.append(data.get("age_mae", 0.0))
        except FileNotFoundError:
            print(f"Advertencia: No se encontro el JSON para {model_key}")
            gender_accuracies.append(0.0)
            age_maes.append(0.0)

    display_names = [BASELINE_DISPLAY_NAME] + list(MODEL_DISPLAY_NAMES)
    gender_accuracies.insert(0, BASELINE_GENDER_ACCURACY)
    age_maes.insert(0, BASELINE_AGE_MAE)

    return display_names, gender_accuracies, age_maes


def _load_loss_history(model_key):
    loss_history_path = METRICS_DIR / f"{model_key}_loss_history.json"
    data = _load_json(loss_history_path)
    return data["train_loss"], data["validation_loss"]


def _load_predictions(model_key):
    metrics_path = METRICS_DIR / f"{model_key}_metrics.json"
    data = _load_json(metrics_path)
    return data["gender_true"], data["gender_pred"], data["age_true"], data["age_pred"]


def plot_model_comparison():
    display_names, gender_accuracies, age_maes = _load_model_metrics()
    x_positions = np.arange(len(display_names))

    fig, ax_mae = plt.subplots(figsize=(12, 6))

    mae_color = "#e74c3c"
    ax_mae.set_xlabel("Experimentos del Laboratorio", fontweight="bold", labelpad=12)
    ax_mae.set_ylabel("Edad - Error Absoluto Medio (MAE en anos)", color=mae_color, fontweight="bold")
    ax_mae.bar(
        x_positions - 0.2,
        age_maes,
        width=0.4,
        color=mae_color,
        alpha=0.7,
        label="MAE Edad (Menor es mejor)",
    )
    ax_mae.tick_params(axis="y", labelcolor=mae_color)
    ax_mae.set_xticks(x_positions)
    ax_mae.set_xticklabels(display_names, rotation=15, ha="right")

    ax_acc = ax_mae.twinx()
    acc_color = "#2c3e50"
    ax_acc.set_ylabel("Genero - Exactitud (Accuracy)", color=acc_color, fontweight="bold")
    ax_acc.plot(
        x_positions + 0.2,
        gender_accuracies,
        color=acc_color,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Acc Genero (Mayor es mejor)",
    )
    ax_acc.tick_params(axis="y", labelcolor=acc_color)

    plt.title(
        "Comparativa Global de Rendimiento: Modelos Clasicos vs Deep Learning (UTKFace)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    output_path = PLOTS_OUTPUT_DIR / "comparativa_modelos_utkface.png"
    plt.savefig(output_path, dpi=300)
    print(f"Grafico '{output_path}' generado con exito.")
    plt.close()


def plot_loss_curves(train_losses, validation_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", marker="o", linewidth=2)
    plt.plot(validation_losses, label="Validation Loss", color="#ff7f0e", linestyle="--", marker="s", linewidth=2)
    plt.title("Curvas de Aprendizaje (ResNet Fine-Tuning)")
    plt.xlabel("Epoch")
    plt.ylabel("Perdida Total (Loss)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    output_path = PLOTS_OUTPUT_DIR / "1_curvas_loss.png"
    plt.savefig(output_path, dpi=300)
    print(f"Curvas de perdida guardadas en '{output_path}'.")
    plt.close()


def plot_gender_confusion_matrix(y_true_gender, y_pred_gender):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_gender, y_pred_gender)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Hombre (0)", "Mujer (1)"],
        yticklabels=["Hombre (0)", "Mujer (1)"],
    )
    plt.title("Matriz de Confusion - Clasificacion de Genero")
    plt.xlabel("Prediccion del Modelo")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    output_path = PLOTS_OUTPUT_DIR / "2_matriz_confusion.png"
    plt.savefig(output_path, dpi=300)
    print(f"Matriz de confusion guardada en '{output_path}'.")
    plt.close()


def plot_age_scatter(y_true_age, y_pred_age):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_age, y_pred_age, alpha=0.4, color="teal", edgecolors="k")

    max_age = max(max(y_true_age), max(y_pred_age))
    plt.plot([0, max_age], [0, max_age], color="red", linestyle="--", linewidth=2, label="Prediccion Perfecta")

    plt.title("Regresion: Edad Real vs. Edad Estimada")
    plt.xlabel("Edad Real (Anos)")
    plt.ylabel("Edad Estimada por el Modelo (Anos)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    output_path = PLOTS_OUTPUT_DIR / "3_edad_scatter.png"
    plt.savefig(output_path, dpi=300)
    print(f"Grafico de dispersion de edades guardado en '{output_path}'.")
    plt.close()


if __name__ == "__main__":
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_model_comparison()

    train_losses, validation_losses = _load_loss_history(RESNET_FINETUNED_KEY)
    plot_loss_curves(train_losses, validation_losses)

    gender_true, gender_pred, age_true, age_pred = _load_predictions(RESNET_FINETUNED_KEY)
    plot_gender_confusion_matrix(gender_true, gender_pred)
    plot_age_scatter(age_true, age_pred)

    print("Todos los graficos exportados en", PLOTS_OUTPUT_DIR)