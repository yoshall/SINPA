import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """
    A module that performs time embedding for input data.

    Args:
        None

    Attributes:
        embed_slot (nn.Embedding): Embedding layer for slot information.
        embed_day (nn.Embedding): Embedding layer for day of the week information.
        embed_util (nn.Embedding): Embedding layer for utilization type information.
        embed_plan (nn.Embedding): Embedding layer for geolocation information.

    Methods:
        forward(x): Performs the forward pass of the time embedding layer.

    """

    def __init__(self):
        super(TimeEmbedding, self).__init__()
        # a day contains 96 15-min slots
        self.embed_slot = nn.Embedding(4 * 24, 3)
        self.embed_day = nn.Embedding(7, 3)  # Day of the Week
        self.embed_util = nn.Embedding(10, 3)  # Utilization Type
        self.embed_plan = nn.Embedding(36, 3)  # Geolocation

    def forward(self, x):
        """
        Performs the forward pass of the time embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, time_steps, num_features].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, time_steps, num_features_embedded].

        """
        # x: slot, day, holiday or not
        x_slot = self.embed_slot(x[..., 0])
        x_day = self.embed_day(x[..., 1])
        x_util = self.embed_util(x[..., 6])
        x_plan = self.embed_plan(x[..., 7])
        out = torch.cat((x_slot, x_day, x[..., 2:6], x_util, x_plan, x[..., 8:]), -1)
        return out  # [b, t, n, 3+3+4+3+3+3=19]
