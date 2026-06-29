import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
SPLIT_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_utkface_name(path):
    parts = Path(path).stem.split("_")

    if len(parts) < 2:
        raise ValueError("El nombre de archivo no tiene el formato esperado de UTKFace.")

    try:
        age = float(parts[0])
        gender = int(parts[1])
    except ValueError as error:
        raise ValueError("No se pudieron extraer edad y genero desde el nombre del archivo.") from error

    return age, gender


train_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class UTKFaceDataset(Dataset):
    def __init__(self, paths, transform=None):
        if isinstance(paths, (str, Path)):
            image_paths = _collect_image_paths(paths)
        else:
            image_paths = [Path(path) for path in paths]

        self.image_paths = image_paths
        self.labels = [parse_utkface_name(path) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        age, gender = self.labels[index]

        with Image.open(path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(gender).long(), torch.tensor(age).float()


def _collect_image_paths(data_dir):
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"No se encontro la carpeta de datos: {root}")

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError("No se encontraron imagenes validas de UTKFace.")

    return image_paths


def _split_lengths(total_count):
    if total_count < 3:
        raise ValueError("Se necesitan al menos tres imagenes para crear las particiones.")

    train_count = int(total_count * 0.70)
    validation_count = int(total_count * 0.15)
    test_count = total_count - train_count - validation_count
    return train_count, validation_count, test_count


def _dataset_from_subset(dataset, subset, transform):
    image_paths = [dataset.image_paths[index] for index in subset.indices]
    return UTKFaceDataset(image_paths, transform=transform)


def get_dataloaders(data_dir):
    set_seed(SPLIT_SEED)
    full_dataset = UTKFaceDataset(data_dir, transform=train_tfms)
    split_lengths = _split_lengths(len(full_dataset))
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_subset, validation_subset, test_subset = random_split(
        full_dataset,
        split_lengths,
        generator=generator,
    )

    train_dataset = _dataset_from_subset(full_dataset, train_subset, train_tfms)
    validation_dataset = _dataset_from_subset(full_dataset, validation_subset, test_tfms)
    test_dataset = _dataset_from_subset(full_dataset, test_subset, test_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, validation_loader, test_loader
