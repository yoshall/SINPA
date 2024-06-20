from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models

    Args:
        name (str): The name of the model.
        dataset: The dataset used for training the model.
        device: The device (e.g., 'cpu', 'cuda') on which the model will be trained.
        num_nodes (int): The number of nodes in the dataset.
        seq_len (int): The length of the input sequence.
        horizon (int): The prediction horizon.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data.
    """

    def __init__(
        self, name, dataset, device, num_nodes, seq_len, horizon, input_dim, output_dim
    ):
        super(BaseModel, self).__init__()
        self.name = name
        self.dataset = dataset
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self):
        """
        Forward pass logic

        Returns:
            Model output
        """
        raise NotImplementedError

    def param_num(self, str):
        """
        Get the number of trainable parameters in the model.

        Args:
            str: A string parameter (not used in the function).

        Returns:
            The total number of trainable parameters in the model.
        """
        return sum([param.nelement() for param in self.parameters()])

    def __str__(self):
        """
        Get a string representation of the model.

        Returns:
            A string representation of the model, including the number of trainable parameters.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
