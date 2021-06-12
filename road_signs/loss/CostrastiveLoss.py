import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, distance_fn):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = self.distance_fn(output1, output2)
        losses = 0.5 * (
            (target.float() * distances.pow(2)) + # 0 or 1
            (1 - target).float() * F.relu(self.margin - (distances + self.eps)).pow(2)
        )

        return losses.mean() if size_average else losses.sum
