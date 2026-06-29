import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.eval import evaluate
from src.models import MultiTaskCNN
from src.models import MultiTaskMLP
from src.models import MultiTaskResNet


DEFAULT_DATA_DIR = Path("data/UTKFace")
DEFAULT_OUTPUT_DIR = Path("outputs/models")
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_LAMBDA_AGE = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena modelos multi-tarea sobre UTKFace.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lambda-age", type=float, default=DEFAULT_LAMBDA_AGE)
    return parser.parse_args()


def build_model_registry():
    return {
        "mlp": lambda: MultiTaskMLP(),
        "cnn": lambda: MultiTaskCNN(),
        "resnet": lambda: MultiTaskResNet(freeze_backbone=True, pretrained=True),
    }


def train_one_epoch(model, loader, optimizer, device, lambda_age):
    gender_loss_function = nn.CrossEntropyLoss()
    age_loss_function = nn.MSELoss()
    total_loss = 0.0

    model.train()

    for images, gender, age in loader:
        images = images.to(device)
        gender = gender.to(device).long().view(-1)
        age = age.to(device).float()

        optimizer.zero_grad()
        gender_output, age_output = model(images)
        age = age.view_as(age_output)
        gender_loss = gender_loss_function(gender_output, gender)
        age_loss = age_loss_function(age_output, age)
        loss = gender_loss + lambda_age * age_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, device, lambda_age):
    gender_loss_function = nn.CrossEntropyLoss()
    age_loss_function = nn.MSELoss()
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for images, gender, age in loader:
            images = images.to(device)
            gender = gender.to(device).long().view(-1)
            age = age.to(device).float()

            gender_output, age_output = model(images)
            age = age.view_as(age_output)
            gender_loss = gender_loss_function(gender_output, gender)
            age_loss = age_loss_function(age_output, age)
            loss = gender_loss + lambda_age * age_loss

            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_checkpoint(model, optimizer, epoch, validation_loss, output_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
        },
        output_path,
    )


def save_metrics(metrics, output_path):
    with open(output_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def print_epoch_summary(model_name, epoch, total_epochs, train_loss, validation_loss):
    print(
        f"[{model_name}] Epoca {epoch + 1}/{total_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {validation_loss:.4f}"
    )


def print_evaluation_summary(model_name, metrics):
    print(f"\n{'=' * 60}")
    print(f"Resultados finales: {model_name}")
    print(f"{'=' * 60}")
    print(f"  Gender Accuracy:  {metrics['gender_accuracy']:.4f}")
    print(f"  Gender F1:        {metrics['gender_f1_weighted']:.4f}")
    print(f"  Age MAE:          {metrics['age_mae']:.2f}")
    print(f"  Age RMSE:         {metrics['age_rmse']:.2f}")
    print(f"  Age R2:           {metrics['age_r2']:.4f}")
    print(f"{'=' * 60}\n")


def train_model(model_name, model, train_loader, validation_loader, test_loader, device, args):
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    best_validation_loss = float("inf")
    checkpoint_path = args.output_dir / f"{model_name}_best.pth"

    print(f"\nEntrenando modelo: {model_name}")
    print(f"Parametros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.lambda_age)
        validation_loss = validate(model, validation_loader, device, args.lambda_age)
        print_epoch_summary(model_name, epoch, args.epochs, train_loss, validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(model, optimizer, epoch, validation_loss, checkpoint_path)

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader)
    print_evaluation_summary(model_name, test_metrics)

    metrics_path = args.output_dir / f"{model_name}_metrics.json"
    save_metrics(test_metrics, metrics_path)

    return test_metrics


def main():
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"No se encontro la carpeta de datos: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_loader, validation_loader, test_loader = get_dataloaders(args.data_dir)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(validation_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model_registry = build_model_registry()
    all_metrics = {}

    for model_name, model_factory in model_registry.items():
        model = model_factory()
        metrics = train_model(
            model_name,
            model,
            train_loader,
            validation_loader,
            test_loader,
            device,
            args,
        )
        all_metrics[model_name] = metrics

    save_metrics(all_metrics, args.output_dir / "all_metrics.json")
    print("Entrenamiento completo. Resultados guardados en:", args.output_dir)


if __name__ == "__main__":
    main()
