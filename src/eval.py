from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from PIL import Image


EXAMPLE_LIMIT = 5
AGE_ERROR_THRESHOLD = 15
OUTPUT_DIRECTORY = Path("outputs/plots_and_csv")


def _format_age(age):
    return f"{age:.1f}".replace("-", "m").replace(".", "p")


def _age_decade(age):
    age_value = max(0, int(np.floor(age)))

    if age_value <= 10:
        return "0-10"

    decade_start = ((age_value - 1) // 10) * 10 + 1
    decade_end = decade_start + 9
    return f"{decade_start}-{decade_end}"


def _age_decade_start(age_range):
    return int(age_range.split("-")[0])


def _age_mae_by_decade(age_true, age_pred):
    age_decades = np.array([_age_decade(age) for age in age_true])
    metrics = {}

    for age_range in sorted(set(age_decades), key=_age_decade_start):
        mask = age_decades == age_range
        metrics[age_range] = float(mean_absolute_error(age_true[mask], age_pred[mask]))

    return metrics


def _image_array(image):
    image_tensor = image.detach().cpu().float()

    if image_tensor.ndim == 3 and image_tensor.shape[0] in (1, 3, 4):
        image_tensor = image_tensor.permute(1, 2, 0)

    if image_tensor.ndim not in (2, 3):
        raise ValueError("La imagen no tiene un formato compatible para guardarse.")

    image_data = image_tensor.numpy()

    if image_data.min() < 0 or image_data.max() > 1:
        value_range = image_data.max() - image_data.min()
        if value_range > 0:
            image_data = (image_data - image_data.min()) / value_range
        else:
            image_data = np.zeros_like(image_data)

    image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)

    if image_data.ndim == 3 and image_data.shape[-1] == 1:
        return image_data[:, :, 0]

    return image_data


def _save_example(
    image,
    output_directory,
    example_type,
    index,
    gender_true,
    gender_pred,
    age_true,
    age_pred,
):
    file_name = (
        f"{example_type}_{index:05d}_"
        f"gender_true_{int(gender_true)}_pred_{int(gender_pred)}_"
        f"age_true_{_format_age(age_true)}_pred_{_format_age(age_pred)}.png"
    )
    file_path = output_directory / file_name
    Image.fromarray(_image_array(image)).save(file_path)
    return str(file_path)


@torch.no_grad()
def evaluate(model, loader):
    was_training = model.training
    device = next(model.parameters()).device
    output_directory = OUTPUT_DIRECTORY

    gender_predictions = []
    gender_targets = []
    age_predictions = []
    age_targets = []
    correct_examples = []
    incorrect_examples = []
    sample_index = 0

    output_directory.mkdir(parents=True, exist_ok=True)
    model.eval()

    try:
        for images, gender_target, age_target in loader:
            model_images = images.to(device)
            gender_output, age_output = model(model_images)

            gender_prediction = torch.argmax(gender_output, dim=1)
            batch_gender_predictions = gender_prediction.detach().cpu().numpy()
            batch_gender_targets = gender_target.reshape(-1).detach().cpu().numpy()
            batch_age_predictions = age_output.reshape(-1).detach().cpu().numpy()
            batch_age_targets = age_target.reshape(-1).detach().cpu().numpy()

            gender_predictions.append(batch_gender_predictions)
            gender_targets.append(batch_gender_targets)
            age_predictions.append(batch_age_predictions)
            age_targets.append(batch_age_targets)

            batch_images = images.detach().cpu()

            for index, image in enumerate(batch_images):
                gender_true = batch_gender_targets[index]
                gender_pred = batch_gender_predictions[index]
                age_true = batch_age_targets[index]
                age_pred = batch_age_predictions[index]
                age_error = abs(age_true - age_pred)
                gender_is_correct = int(gender_true) == int(gender_pred)
                age_is_correct = age_error <= AGE_ERROR_THRESHOLD

                if gender_is_correct and age_is_correct and len(correct_examples) < EXAMPLE_LIMIT:
                    file_path = _save_example(
                        image,
                        output_directory,
                        "correct",
                        sample_index,
                        gender_true,
                        gender_pred,
                        age_true,
                        age_pred,
                    )
                    correct_examples.append(file_path)

                if (not gender_is_correct or not age_is_correct) and len(incorrect_examples) < EXAMPLE_LIMIT:
                    file_path = _save_example(
                        image,
                        output_directory,
                        "incorrect",
                        sample_index,
                        gender_true,
                        gender_pred,
                        age_true,
                        age_pred,
                    )
                    incorrect_examples.append(file_path)

                sample_index += 1
    finally:
        if was_training:
            model.train()

    if not gender_targets:
        raise ValueError("El loader no contiene muestras para evaluar.")

    y_gender_pred = np.concatenate(gender_predictions).astype(int)
    y_gender_true = np.concatenate(gender_targets).astype(int)
    y_age_pred = np.concatenate(age_predictions).astype(float)
    y_age_true = np.concatenate(age_targets).astype(float)

    gender_accuracy = accuracy_score(y_gender_true, y_gender_pred)
    gender_f1 = f1_score(
        y_gender_true,
        y_gender_pred,
        average="weighted",
        zero_division=0,
    )
    gender_precision = precision_score(
        y_gender_true,
        y_gender_pred,
        average="weighted",
        zero_division=0,
    )
    gender_recall = recall_score(
        y_gender_true,
        y_gender_pred,
        average="weighted",
        zero_division=0,
    )
    gender_confusion_matrix = confusion_matrix(
        y_gender_true,
        y_gender_pred,
        labels=[0, 1],
    ).tolist()

    age_mae = mean_absolute_error(y_age_true, y_age_pred)
    age_rmse = np.sqrt(mean_squared_error(y_age_true, y_age_pred))
    age_r2 = r2_score(y_age_true, y_age_pred)
    age_decade_mae = _age_mae_by_decade(y_age_true, y_age_pred)

    return {
        "gender_accuracy": float(gender_accuracy),
        "gender_f1_weighted": float(gender_f1),
        "gender_precision_weighted": float(gender_precision),
        "gender_recall_weighted": float(gender_recall),
        "gender_confusion_matrix": gender_confusion_matrix,
        "age_mae": float(age_mae),
        "age_rmse": float(age_rmse),
        "age_r2": float(age_r2),
        "age_mae_by_decade": age_decade_mae,
        "correct_examples_saved": correct_examples,
        "incorrect_examples_saved": incorrect_examples,
        "correct_examples_count": len(correct_examples),
        "incorrect_examples_count": len(incorrect_examples),
    }
