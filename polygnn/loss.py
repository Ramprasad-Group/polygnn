from torch import nn
import torch
import numpy as np
import random

# fix random seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


class mt_loss(nn.Module):
    def __init__(self):
        super(mt_loss, self).__init__()

    def forward(self, predictions, data):
        squared_error = (predictions - data.y).abs().square()

        return ((squared_error * data.selector).sum() / data.selector.sum()).sqrt()


class st_loss(nn.Module):
    def __init__(self):
        super(st_loss, self).__init__()
        self.mse_fn = nn.MSELoss()

    def forward(self, predictions, data):
        # enforce the right shapes
        predictions = predictions.view(
            data.num_graphs,
        )
        data.y = data.y.view(
            data.num_graphs,
        )
        # ########################
        mse = self.mse_fn(predictions, data.y)

        return mse
