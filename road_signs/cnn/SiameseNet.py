import torch.nn as nn
from road_signs.cnn.RoadSignNet import RoadSignNet


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn_net = RoadSignNet()

    def forward(self, x1, x2):
        output1 = self.cnn_net(x1)
        output2 = self.cnn_net(x2)
        return output1, output2
