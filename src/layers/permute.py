from torch import nn, Tensor


class Permute(nn.Module):
    """
    A module that permutes the dimensions of a tensor.

    Args:
        *args: The desired order of dimensions for permutation.

    Shape:
        - Input: (batch_size, ..., dim1, dim2, ..., dimN)
        - Output: (batch_size, ..., dimP[0], dimP[1], ..., dimP[N])

    Examples:
        >>> permute = Permute(2, 0, 1)
        >>> x = torch.randn(10, 3, 5)
        >>> output = permute(x)
        >>> print(output.shape)
        torch.Size([5, 10, 3])
    """

    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, x: Tensor):
        """
        Forward pass of the Permute module.

        Args:
            x (torch.Tensor): The input tensor to be permuted.

        Returns:
            torch.Tensor: The permuted tensor.

        Shape:
            - Input: (batch_size, ..., dim1, dim2, ..., dimN)
            - Output: (batch_size, ..., dimP[0], dimP[1], ..., dimP[N])
        """
        return x.permute(self.dims)
