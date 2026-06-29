import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, lambda_age=0.01):
    device = torch.device("cuda")
    gender_loss_function = nn.CrossEntropyLoss()
    age_loss_function = nn.MSELoss()
    total_loss = 0.0

    model.to(device)
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
