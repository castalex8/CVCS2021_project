import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class PreTrainedRoadSignNet(nn.Module):
    def __init__(self, weights='weigths/ClassificationWeights/weightsNoSoftmax3layers.pth'):
        super().__init__()
        self.load_state_dict(torch.load(weights))
        self.eval()


class PreTrainedTripletNet(ResNet):
    def __init__(self):
        super(PreTrainedTripletNet, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        x1, x2, x3 = x
        output1 = super().forward(x1)
        output2 = super().forward(x2)
        output3 = super().forward(x3)
        return output1, output2, output3
