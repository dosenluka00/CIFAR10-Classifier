import argparse

from config import (
    MODEL_NAME,
    EPOCHS
)

from train_model import run_training


parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default=MODEL_NAME,
    choices=["random", "mlp", "cnn"]
)

parser.add_argument(
    "--epochs",
    type=int,
    default=EPOCHS
)

parser.add_argument(
    "--lr",
    type=float,
    default=None
)

args = parser.parse_args()

run_training(
    model_name=args.model,
    epochs=args.epochs,
    learning_rate=args.lr
)