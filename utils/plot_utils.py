import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


def plot_training_history(
    train_losses,
    val_accuracies,
    output_dir="outputs"
):
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()

    plt.plot(epochs, train_losses)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(
        f"{output_dir}/loss_curve.png"
    )

    plt.close()

    # Accuracy plot
    plt.figure()

    plt.plot(epochs, val_accuracies)

    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.savefig(
        f"{output_dir}/accuracy_curve.png"
    )

    plt.close()

def plot_confusion_matrix(
    labels,
    predictions,
    class_names,
    output_dir="outputs"
):
    cm = confusion_matrix(
        labels,
        predictions
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=True
    )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()