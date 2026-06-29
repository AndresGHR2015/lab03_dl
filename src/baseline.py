import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena la linea base clasica para UTKFace.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/UTKFace"))
    return parser.parse_args()


def extract_flat_data(loader):
    features = []
    gender_targets = []
    age_targets = []

    for images, gender, age in loader:
        batch_size = images.size(0)
        features.append(images.view(batch_size, -1).numpy())
        gender_targets.append(gender.view(batch_size).numpy())
        age_targets.append(age.view(batch_size).numpy())

    if not features:
        raise ValueError("El cargador no contiene muestras para extraer datos.")

    X_flat = np.concatenate(features, axis=0)
    y_gender = np.concatenate(gender_targets, axis=0)
    y_age = np.concatenate(age_targets, axis=0)
    return X_flat, y_gender, y_age


def get_pca_components(X_train_flat):
    return min(100, X_train_flat.shape[0], X_train_flat.shape[1])


def build_gender_pipeline(n_components):
    return Pipeline(
        [
            ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
            ("clf", GaussianNB()),
        ]
    )


def build_age_pipeline(n_components):
    return Pipeline(
        [
            ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def evaluate_gender_pipeline(pipeline, X_test_flat, y_test):
    predictions = pipeline.predict(X_test_flat)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_age_pipeline(pipeline, X_test_flat, y_test):
    predictions = pipeline.predict(X_test_flat)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }


def print_metrics(gender_metrics, age_metrics):
    print(f"Exactitud de genero: {gender_metrics['accuracy']:.4f}")
    print("Reporte de clasificacion de genero:")
    print_classification_report(gender_metrics["classification_report"])
    print(f"MAE de edad: {age_metrics['mae']:.4f}")
    print(f"RMSE de edad: {age_metrics['rmse']:.4f}")
    print(f"R2 de edad: {age_metrics['r2']:.4f}")


def print_classification_report(report):
    for label, metrics in report.items():
        if label == "accuracy":
            print(f"Exactitud global: {metrics:.4f}")
            continue

        display_label = {
            "macro avg": "Promedio macro",
            "weighted avg": "Promedio ponderado",
        }.get(label, f"Clase {label}")
        print(
            f"{display_label}: "
            f"precision={metrics['precision']:.4f}, "
            f"sensibilidad={metrics['recall']:.4f}, "
            f"f1={metrics['f1-score']:.4f}, "
            f"soporte={metrics['support']:.0f}"
        )


def main():
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"No se encontro la carpeta de datos: {args.data_dir}")

    train_loader, _, test_loader = get_dataloaders(args.data_dir)
    X_train_flat, y_gender_train, y_age_train = extract_flat_data(train_loader)
    X_test_flat, y_gender_test, y_age_test = extract_flat_data(test_loader)

    n_components = get_pca_components(X_train_flat)
    gender_pipeline = build_gender_pipeline(n_components)
    age_pipeline = build_age_pipeline(n_components)

    gender_pipeline.fit(X_train_flat, y_gender_train)
    age_pipeline.fit(X_train_flat, y_age_train)

    gender_metrics = evaluate_gender_pipeline(
        gender_pipeline,
        X_test_flat,
        y_gender_test,
    )
    age_metrics = evaluate_age_pipeline(
        age_pipeline,
        X_test_flat,
        y_age_test,
    )
    print_metrics(gender_metrics, age_metrics)


if __name__ == "__main__":
    main()
