import torch.nn as nn

##input: 3 x 32 y 32
## after first convolution and pooling it becomes 32 x 16 x 16
## after second -//- 64 x 8 x 8
## flatten, linear(4096 ->128)
## linear (128 -> 10 output classes)

##pros: edges, textures, object parts recognition, fast
##cons: does not recognize spatial structure

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x