import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Classifier, self).__init__()
        # classification layers (for scores)
        self.classification_layer_1 = nn.Linear(input_channel, 15)
        self.classification_layer_2 = nn.Linear(15, 15)
        self.classification_layer_3 = nn.LayerNorm(15, 2)

        # regression layers (for bounding boxes)
        self.regression_layer_1 = nn.Linear(input_channel, 15)
        self.regression_layer_2 = nn.Linear(15, 15)
        self.regression_layer_3 = nn.LayerNorm(15, num_classes * 4)

    def forward(self, x):
        scores = self.classification_layer_3(
            self.classification_layer_2(self.classification_layer_1(x))
        )
        bbox_coord = self.regression_layer_3(
            self.regression_layer_2(self.regression_layer_1(x))
        )
        return scores, bbox_coord
