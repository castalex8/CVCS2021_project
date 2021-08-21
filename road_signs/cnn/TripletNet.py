import torch.nn as nn
from road_signs.cnn.RoadSignNet import RoadSignNet


class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.cnn_net = RoadSignNet(is_retrieval=True)

    def forward(self, x1, x2, x3):
        output1 = self.cnn_net(x1)
        output2 = self.cnn_net(x2)
        output3 = self.cnn_net(x3)
        return output1, output2, output3
