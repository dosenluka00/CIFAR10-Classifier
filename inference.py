import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import random

from models.cnn import CNNClassifier

import matplotlib.pyplot as plt
import numpy as np


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Transform
transform = transforms.ToTensor()

# Test dataset
test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

# Load model
model = CNNClassifier()

model.load_state_dict(
    torch.load(
        "outputs/cnn_best.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Pick image
index = random.randint(
    0,
    len(test_dataset) - 1
)

image, label = test_dataset[index]

# Add batch dimension
image = image.unsqueeze(0).to(device)

# Prediction
with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        1
    )

predicted_class = test_dataset.classes[
    prediction.item()
]

true_class = test_dataset.classes[
    label
]

print(f"Image index: {index}")
print(f"True class: {true_class}")
print(f"Predicted class: {predicted_class}")
print(
    f"Confidence: "
    f"{confidence.item()*100:.2f}%"
)

# Convert tensor to image
display_image = image.squeeze(0).cpu()

display_image = display_image.permute(
    1, 2, 0
)

display_image = display_image.numpy()

# Display image
plt.imshow(display_image)

plt.title(
    f"True: {true_class}\n"
    f"Pred: {predicted_class}\n"
    f"Conf: {confidence.item()*100:.2f}%"
)

plt.axis("off")

plt.show()