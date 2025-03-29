import math

import torch
import numpy as np

def accuracy(z, targets):
    pred_all = (z > math.log(0.5)).float()
    return torch.mean((pred_all == targets).float()).item()



def euclidean_distance(X, v):
    return torch.sum(torch.square(X - v), dim=-1)
