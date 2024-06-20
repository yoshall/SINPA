from torch import nn, Tensor


class Reshape(nn.Module):
    """
    Reshape module that reshapes the input tensor to the specified shape.

    Args:
        *args: Variable length argument representing the desired shape of the output tensor.

    Attributes:
        shape: The desired shape of the output tensor.

    """

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x: Tensor):
        """
        Forward pass of the Reshape module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The reshaped tensor.

        """
        return x.reshape(self.shape)
