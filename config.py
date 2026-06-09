BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50 ## ok since we have early stopping
RANDOM_SEED = 42

## optins are: cnn, mlp, random
MODEL_NAME = "cnn"

EARLY_STOPPING_PATIENCE = 5

LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5