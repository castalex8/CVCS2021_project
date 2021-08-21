import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    https://towardsdatascience.com/how-to-choose-your-loss-when-designing-a-siamese-neural-net-contrastive-triplet-or-quadruplet-ecba11944ec
    """

    def __init__(self, margin, distance_fn=CosineSimilarity()):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.eps = 1e-9

    def forward(self, output1, output2, target):
        distances = self.distance_fn(output1, output2)
        losses = 0.5 * (
            (target.float() * distances.pow(2)) +
            (1 - target).float() * F.relu(self.margin - (distances + self.eps)).pow(2)
        )

        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()
