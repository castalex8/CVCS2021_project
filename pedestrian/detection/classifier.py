import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Classifier, self).__init__()
        # classification layers (for scores)
        self.classification = nn.Sequential(
            nn.Linear(input_channel, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.LayerNorm(15, 2)
        )
        # regression layers (for bounding boxes)
        self.regression = nn.Sequential(
            nn.Linear(input_channel, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.LayerNorm(15, num_classes * 4)
        )

    def forward(self, x):
        scores = self.classification(x)
        bbox_coord = self.regression(x)
        return scores, bbox_coord
