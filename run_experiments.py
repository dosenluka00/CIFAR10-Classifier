from train_model import run_training
import csv
from config import EPOCHS

models = [
    "random",
    "mlp",
    "cnn"
]

results = []

for model_name in models:

    print("\n" + "=" * 60)
    print(f"Running experiment: {model_name}")
    print("=" * 60)

    result = run_training(model_name=model_name, epochs=EPOCHS)

    results.append(result)

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

for result in results:

    print(
        f"{result['model']:10} | "
        f"Params: {result['parameters']:>10,} | "
        f"Val: {result['best_val_accuracy']:.2f}% | "
        f"Test: {result['test_accuracy']:.2f}%"
    )

with open(
    "outputs/comparison.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "model",
        "parameters",
        "val_accuracy",
        "test_accuracy"
    ])

    for result in results:

        writer.writerow([
            result["model"],
            result["parameters"],
            round(
                result["best_val_accuracy"],
                2
            ),
            round(
                result["test_accuracy"],
                2
            )
        ])

print(
    "\nComparison table saved to outputs/comparison.csv"
)