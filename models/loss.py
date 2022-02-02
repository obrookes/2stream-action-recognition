import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kornia.losses import FocalLoss

weights = None

# Return loss function as required by config
def initialise_loss(loss):
    if(loss =='focal'):
        return FocalLoss(alpha=1, gamma=1, reduction="mean")

    if(loss == "cross_entropy"):
        if loss['cross_entropy']['weighted']:
            return nn.CrossEntropyLoss(weight=weights)
        else:
            return nn.CrossEntropyLoss()

# Compute weights for weighted cross entropy if required
def initialise_ce_weights(class_sample_count):
    largest_class_size = max(class_sample_count)
    weights = torch.empty(len(class_sample_count), dtype=torch.float)
    for i in range(0, len(class_sample_count)):
        weights[i] = largest_class_size / class_sample_count[i]
