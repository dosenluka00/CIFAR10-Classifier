import torch
import torch.nn as nn
import torch.optim as optim

import json
from datetime import datetime

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Subset

from models.mlp import MLPClassifier

from utils.report_utils import (
    generate_classification_report
)
from utils.train_utils import (
    train_one_epoch,
    evaluate,
    get_predictions
)

from models.cnn import CNNClassifier
from models.mlp import MLPClassifier
from models.random_classifier import RandomClassifier

import os

from utils.plot_utils import (
    plot_training_history,
    plot_confusion_matrix
)

from utils.model_utils import count_parameters



from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED,
    MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR
)

def run_training(model_name, epochs, learning_rate=None):

    import time

    start_time = time.time()

    # Reproducibility
    torch.manual_seed(RANDOM_SEED)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    output_dir = f"outputs/{model_name}"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Transform, mean and std are cifar10 relevant statistics, data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4
        ),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # Dataset
    ## augmented -> train_transform
    ## clean -> test_transform
    full_train_dataset_augmented = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_dataset_clean = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=test_transform
    )
    ## test
    test_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # Splitting
    train_size = int(
        0.8 * len(full_train_dataset_augmented)
    )

    val_size = len(
        full_train_dataset_augmented
    ) - train_size

    ## deterministic train/val indices
    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    train_indices, val_indices = random_split(
        range(len(full_train_dataset_augmented)),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = Subset(
        full_train_dataset_augmented,
        train_indices.indices
    )

    val_dataset = Subset(
        full_train_dataset_clean,
        val_indices.indices
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Model
    if model_name == "random":
        model = RandomClassifier()

    elif model_name == "mlp":
        model = MLPClassifier()

    elif model_name == "cnn":
        model = CNNClassifier()

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model = model.to(device)

    print(f"Training model: {model_name}")
    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    ##since random guesser does not train, we need the following:
    if model_name == "random":

        test_accuracy = evaluate(
            model,
            test_loader,
            device
        )

        print(f"Random Baseline Accuracy: {test_accuracy:.2f}%")

        return {
        "model": model_name,
        "parameters": count_parameters(model),
        "best_val_accuracy": 0.0,
        "test_accuracy": test_accuracy
        }

    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer

    lr = (
        learning_rate
        if learning_rate is not None
        else LEARNING_RATE
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE
    )

    ##for future plots
    train_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    epochs_without_improvement = 0





    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        val_accuracy = evaluate(
            model,
            val_loader,
            device
        )
        ## for learning rate (ReduceLROnPlateau)
        scheduler.step(val_accuracy)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                f"{output_dir}/{model_name}_best.pth"
            )
        else:

            epochs_without_improvement += 1

        if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE):

            print(
                f"\nEarly stopping triggered "
                f"(patience={EARLY_STOPPING_PATIENCE}) "
                f"after {epoch + 1} epochs."
            )

            break

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    model.load_state_dict(
        torch.load(
            f"{output_dir}/{model_name}_best.pth",
            map_location=device
            )
    )

    test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    y_true, y_pred = get_predictions(
        model,
        test_loader,
        device
    )

    report = generate_classification_report(
        y_true,
        y_pred,
        test_dataset.classes
    )




    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    print("\nClassification Report:\n")
    print(report)

    with open(
        f"{output_dir}/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    print(
        "Classification report saved to outputs."
    )

    training_time = time.time() - start_time
    print(
    f"Training completed in "
    f"{training_time / 60:.2f} minutes"
    )

    results = {
        "model": model_name,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "learning_rate": lr,
        "best_validation_accuracy": round(best_val_accuracy, 2),
        "test_accuracy": round(test_accuracy, 2),
        "training_time_seconds": round(training_time, 2)
    }
    with open(
        f"{output_dir}/results.json",
        "w"
    ) as f:
        
        json.dump(results, f, indent=4)

    print("Results JSON saved")
    

    labels, predictions = get_predictions(
        model,
        test_loader,
        device
    )

    plot_confusion_matrix(
        labels,
        predictions,
        test_dataset.classes,
        output_dir
    )

    print("Confusion matrix saved.")

    plot_training_history(
        train_losses,
        val_accuracies,
        output_dir
    )

    print("Plots saved to outputs/")

    return {
    "model": model_name,
    "parameters": count_parameters(model),
    "best_val_accuracy": best_val_accuracy,
    "test_accuracy": test_accuracy
    }
