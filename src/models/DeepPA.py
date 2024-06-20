import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from src.base.model import BaseModel
from src.layers.embedding import TimeEmbedding
import numpy as np
import time


class DeepPA(BaseModel):
    """
    DeepPA model for time series prediction.

    Args:
        dropout (float): Dropout rate (default: 0.3).
        spatial_flag (bool): Whether to use Spatial Transformer (default: True).
        temporal_flag (bool): Whether to use Temporal Transformer (default: True).
        spatial_encoding (bool): Whether to use spatial encoding (default: True).
        temporal_encoding (bool): Whether to use temporal encoding (default: True).
        temporal_PE (bool): Whether to use temporal positional encoding (default: True).
        GCO (bool): Whether to use Graph Convolution Operator (default: True).
        CLUSTER (bool): Whether to use clustering (default: True).
        n_hidden (int): Number of hidden units (default: 32).
        end_channels (int): Number of output channels in the end convolutional layers (default: 512).
        n_blocks (int): Number of blocks in the model (default: 2).
        n_heads (int): Number of attention heads (default: 2).
        mlp_expansion (int): Expansion factor for the MLP layers (default: 2).
        covar_dim (int): Dimension of the covariate input (default: 10).
        GCO_Thre (float): Threshold for Graph Convolution Operator (default: 0.5).
        **args: Additional keyword arguments.

    Attributes:
        dropout (float): Dropout rate.
        n_blocks (int): Number of blocks in the model.
        spatial_flag (bool): Whether to use Spatial Transformer.
        temporal_flag (bool): Whether to use Temporal Transformer.
        spatial_encoding (bool): Whether to use spatial encoding.
        temporal_encoding (bool): Whether to use temporal encoding.
        temporal_PE (bool): Whether to use temporal positional encoding.
        GCO (bool): Whether to use Graph Convolution Operator.
        CLUSTER (bool): Whether to use clustering.
        GCO_Thre (float): Threshold for Graph Convolution Operator.
        assignment (torch.Tensor): Assignment matrix for spatial encoding.
        mask (torch.Tensor): Mask matrix for spatial encoding.
        t_modules (nn.ModuleList): List of TemporalTransformer modules.
        s_modules (nn.ModuleList): List of SpatialTransformer modules.
        temporal_convs (nn.ModuleList): List of temporal convolutional layers.
        spatial_convs (nn.ModuleList): List of spatial convolutional layers.
        skip_convs (nn.ModuleList): List of skip connection convolutional layers.
        embed (TimeEmbedding): Time embedding module.
        start_conv (nn.Conv2d): Start convolutional layer.
        covar_linear (nn.Sequential): Covariate linear layers.
        end_conv_1 (nn.Conv2d): First end convolutional layer.
        end_conv_2 (nn.Conv2d): Second end convolutional layer.
    """

    def __init__(
        self,
        dropout=0.3,
        spatial_flag=True,
        temporal_flag=True,
        spatial_encoding=True,
        temporal_encoding=True,
        temporal_PE=True,
        GCO=True,
        CLUSTER=True,
        n_hidden=32,
        end_channels=512,
        n_blocks=2,
        n_heads=2,
        mlp_expansion=2,
        covar_dim=10,
        GCO_Thre=0.5,
        **args,
    ):
        super(DeepPA, self).__init__(**args)
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.spatial_flag = spatial_flag
        self.temporal_flag = temporal_flag
        self.spatial_encoding = spatial_encoding
        self.temporal_encoding = temporal_encoding
        self.temporal_PE = temporal_PE
        self.GCO = GCO
        self.CLUSTER = CLUSTER
        self.GCO_Thre = GCO_Thre

        path_assignment = "data/region/assignment.npy"  # [n, m]
        path_mask = "data/region/mask.npy"  # [n, n]
        self.assignment = (
            torch.from_numpy(np.load(path_assignment)).float().to(self.device)
        )
        self.mask = torch.from_numpy(np.load(path_mask)).bool().to(self.device)
        dist = torch.Tensor(np.load("data/base/dist.npy")).to(self.device)
        dist_mask = dist > 1
        self.mask = torch.logical_and(self.mask, dist_mask)

        if not self.temporal_flag:
            self.temporal_convs = nn.ModuleList()
        else:
            self.t_modules = nn.ModuleList()

        if not self.spatial_flag:
            self.spatial_convs = nn.ModuleList()
        else:
            self.s_modules = nn.ModuleList()

        self.skip_convs = nn.ModuleList()
        self.embed = TimeEmbedding()
        self.start_conv = nn.Conv2d(
            in_channels=1, out_channels=n_hidden, kernel_size=(1, 3), padding=(0, 1)
        )
        self.covar_linear = nn.Sequential(
            nn.Linear(covar_dim, n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_hidden // 2, n_hidden),
        )

        for _ in range(n_blocks):
            window_size = self.seq_len
            print("ws=", window_size)
            if self.temporal_flag:
                self.t_modules.append(
                    TemporalTransformer(
                        temporal_PE,
                        n_hidden,
                        depth=1,
                        heads=n_heads,
                        window_size=window_size,
                        mlp_dim=n_hidden * mlp_expansion,
                        num_time=self.seq_len,
                        device=self.device,
                    )
                )
            else:
                self.temporal_convs.append(
                    nn.Conv1d(
                        in_channels=n_hidden, out_channels=n_hidden, kernel_size=(1, 1)
                    )
                )
            if self.spatial_flag:
                self.s_modules.append(
                    SpatialTransformer(
                        spatial_encoding,
                        temporal_encoding,
                        n_hidden,
                        GCO=self.GCO,
                        CLUSTER=self.CLUSTER,
                        depth=1,
                        heads=n_heads,
                        mlp_dim=n_hidden * mlp_expansion,
                        num_nodes=self.num_nodes,
                        assignment=self.assignment,
                        mask=self.mask,
                        dropout=dropout,
                        device=self.device,
                        GCO_Thre=self.GCO_Thre,
                    )
                )
            else:
                self.spatial_convs.append(
                    nn.Conv1d(
                        in_channels=n_hidden, out_channels=n_hidden, kernel_size=(1, 1)
                    )
                )

        self.end_conv_1 = nn.Conv2d(
            in_channels=n_hidden,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.horizon * self.output_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, X, supports=None):
        """
        Forward pass of the DeepPA model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_nodes, input_dim).
            supports (list): List of support tensors for spatial encoding (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, horizon, num_nodes, output_dim).

        """
        x_embed = self.embed(X[..., 1:].long())
        X = torch.cat((X[..., 0:1], x_embed), -1)  # [b, t, n, c]
        X = X[..., 0:1]
        covars = x_embed[:, :, 0, :-9]  # [b, t, 10]
        semantic = x_embed[0, 0, :, -9:]  # [n, 9],

        B, T, N, C = X.shape

        x = X.permute(0, 3, 2, 1)  # [b, c, n, t]
        x = self.start_conv(x)
        covars = (
            self.covar_linear(covars).unsqueeze(2).permute(0, 3, 2, 1)
        )  # [b, c, 1, t] [8, 64, 1, 12]

        for i in range(self.n_blocks):
            if self.spatial_flag:
                x, covars = self.s_modules[i](
                    x, supports[0], semantic, covars
                )  # [b, c, n, t] [b, c, 1, t]
            else:
                x = self.spatial_convs[i](x)

            if self.temporal_flag:
                x, covars = self.t_modules[i](x, covars)  # [b, c, n, t]
            else:
                x = self.temporal_convs[i](x)
            # x = self.bn[i](x)   # [b, c, n, t]

        x_hat = F.relu(self.end_conv_1(x[..., -1:]))
        x_hat = self.end_conv_2(x_hat).reshape(B, self.horizon, self.output_dim, N)
        x_hat = x_hat.permute(0, 1, 3, 2)

        return x_hat  # [b, t, n, c]


def pair(t):
    """
    Returns a tuple with two elements.

    If the input `t` is already a tuple, it is returned as is.
    If the input `t` is not a tuple, it is wrapped in a tuple and returned.

    Args:
        t: The input value.

    Returns:
        tuple: A tuple with two elements.

    Examples:
        >>> pair(3)
        (3, 3)

        >>> pair((1, 2))
        (1, 2)
    """
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """
    Pre-normalization module that applies layer normalization before forwarding the input to the given function.

    Args:
        dim (int): The input dimension.
        fn (callable): The function to be applied after layer normalization.

    Attributes:
        norm (nn.LayerNorm): The layer normalization module.
        fn (callable): The function to be applied after layer normalization.

    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the PreNorm module.

        Args:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The output tensor after applying layer normalization and the given function.

        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    A feedforward neural network module.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The dimension of the hidden layer.
        dropout (float, optional): The dropout probability. Default is 0.0.

    Attributes:
        net (nn.Sequential): The sequential network architecture.

    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass of the feedforward neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.net(x)


class GCO_Module(nn.Module):
    """
    GCO_Module (Graph Convolution Operator) module.

    Args:
        hidden_size (int): The size of the hidden state.
        num_blocks (int): The number of blocks to divide the hidden state into.
        GCO_Thre (int, optional): The threshold for the number of modes to keep. Defaults to 1.
        hidden_size_factor (int, optional): The factor to scale the hidden size by. Defaults to 1.
    """

    def __init__(self, hidden_size, num_blocks, GCO_Thre=1, hidden_size_factor=1):
        super().__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.GCO_Thre = GCO_Thre
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks, self.block_size)
        )

    def forward(self, x):
        """
        Forward pass of the GCO module.

        Args:
            x (torch.Tensor): The input tensor of shape (B, N, C), where B is the batch size,
                N is the sequence length, and C is the number of channels.

        Returns:
            torch.Tensor: The output tensor of shape (B, N, C), representing the result of the
                GCO operation applied to the input tensor.
        """
        bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        x = torch.fft.rfft(x, dim=1, norm="ortho")

        x = x.reshape(B, N // 2 + 1, self.num_blocks, self.block_size)

        real_1 = torch.zeros(
            [B, N // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        real_2 = torch.zeros(x.shape, device=x.device)
        imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.GCO_Thre)

        real_1[:, :kept_modes] = F.relu(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1) + self.b1
        )

        real_2[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", real_1[:, :kept_modes], self.w2) + self.b2
        )

        x = torch.stack([real_2, imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, N // 2 + 1, C)
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = x.type(dtype)
        return x + bias


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer module for DeepPA model.

    Args:
        spatial_encoding (bool): Flag indicating whether to use spatial encoding.
        temporal_encoding (bool): Flag indicating whether to use temporal encoding.
        dim (int): Dimension of the input features.
        GCO (nn.Module): Graph Convolutional Operator module.
        CLUSTER (nn.Module): Cluster module.
        depth (int): Number of layers in the model.
        heads (int): Number of attention heads.
        device: Device to be used for computation.
        mlp_dim (int): Dimension of the feedforward network in the model.
        num_nodes (int): Number of nodes in the graph.
        assignment: Assignment module.
        mask: Mask module.
        GCO_Thre: Threshold for Graph Convolutional Operator.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        semantic_dim (int, optional): Dimension of the semantic features. Defaults to 9.
        covar_dim (int, optional): Dimension of the covariate features. Defaults to 10.
        cluster (int, optional): Number of clusters. Defaults to 100.
        attn_scale (float, optional): Scaling factor for attention weights. Defaults to 0.01.
    """

    def __init__(
        self,
        spatial_encoding,
        temporal_encoding,
        dim,
        GCO,
        CLUSTER,
        depth,
        heads,
        device,
        mlp_dim,
        num_nodes,
        assignment,
        mask,
        GCO_Thre,
        dropout=0.0,
        semantic_dim=9,
        covar_dim=10,
        cluster=100,
        attn_scale=0.01,
    ):
        super().__init__()
        self.spatial_encoding = spatial_encoding
        self.temporal_encoding = temporal_encoding
        self.GCO = GCO
        self.CLUSTER = CLUSTER
        self.GCO_Thre = GCO_Thre

        self.attn_scale = attn_scale

        if self.spatial_encoding:
            self.sematic_to_embedding = nn.Sequential(
                nn.Linear(semantic_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, dim - dim // 8),
            )
            self.pos_embedding = nn.Parameter(torch.randn(num_nodes, dim // 8))

        self.layers = nn.ModuleList([])
        SA_node = num_nodes
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        GCO_Module(
                            hidden_size=dim,
                            num_blocks=8,
                            hard_thresholding_fraction=1,
                            hidden_size_factor=1,
                            GCO_Thre=self.GCO_Thre,
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, adj, semantic, covars=None):
        """
        Forward pass of the SpatialTransformer module.

        Args:
            x (torch.Tensor): Input tensor of shape [b, c, n, t].
            adj: Adjacency matrix.
            semantic (torch.Tensor): Semantic tensor of shape [n, 10].
            covars (torch.Tensor, optional): Covariate tensor of shape [b, c, 1, t]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [b, c, n+1 or n, t] if temporal encoding is used, otherwise [b, c, n, t].
            torch.Tensor: Covariate tensor of shape [b, c, 1, t] if temporal encoding is used, otherwise None.
        """
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        covars_tmp = covars.permute(0, 3, 2, 1).reshape(b * t, 1, c)

        if self.spatial_encoding:
            x = x + torch.cat(
                [self.pos_embedding, self.sematic_to_embedding(semantic)], dim=-1
            )
            pos_embed = self.pos_embedding
            sematic_to_embed = self.sematic_to_embedding(semantic)

        if self.temporal_encoding:
            x = torch.cat([covars_tmp, x], dim=1)
            n = n + 1
        else:
            n = n

        for gco, ff in self.layers:
            if self.gco:
                x = gco(x) + x
            x = ff(x) + x

        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        if self.temporal_encoding:
            return x[:, :, 1:, :], x[:, :, :1, :]
        else:
            return x, covars


class TemporalAttention(nn.Module):
    """
    Temporal Attention module that applies self-attention mechanism over the temporal dimension of the input.

    Args:
        dim (int): The input feature dimension.
        heads (int, optional): The number of attention heads. Defaults to 2.
        window_size (int, optional): The size of the attention window. Defaults to 1.
        qkv_bias (bool, optional): Whether to include bias terms in the query, key, and value linear layers. Defaults to False.
        qk_scale (float, optional): Scale factor for the query and key. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        causal (bool, optional): Whether to apply causal masking to the attention weights. Defaults to True.
        device (torch.device, optional): The device on which the module is located. Defaults to None.

    Attributes:
        dim (int): The input feature dimension.
        num_heads (int): The number of attention heads.
        causal (bool): Whether to apply causal masking to the attention weights.
        window_size (int): The size of the attention window.
        scale (float): Scale factor for the query and key.
        qkv (nn.Linear): Linear layer for the query, key, and value projection.
        attn_drop (nn.Dropout): Dropout layer for attention weights.
        proj (nn.Linear): Linear layer for the output projection.
        proj_drop (nn.Dropout): Dropout layer for the output.

    Methods:
        forward(x): Performs a forward pass of the TemporalAttention module.

    """

    def __init__(
        self,
        dim,
        heads=2,
        window_size=1,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        causal=True,
        device=None,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        self.window_size = window_size
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)

    def forward(self, x):
        """
        Performs a forward pass of the TemporalAttention module.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C is the input feature dimension.

        Returns:
            torch.Tensor: The output tensor of shape (B, T, C), where B is the batch size, T is the sequence length, and C is the output feature dimension.

        """
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # merge key padding and attention masks
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [b, heads, T, T]

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class TemporalTransformer(nn.Module):
    """
    TemporalTransformer module that applies temporal attention and feed-forward layers.

    Args:
        temporal_PE (bool): Whether to use temporal positional encoding.
        dim (int): Dimension of the input tensor.
        depth (int): Number of layers in the transformer.
        heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mlp_dim (int): Dimension of the feed-forward layer.
        num_time (int): Number of time steps.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        device (str, optional): Device to run the module on. Defaults to None.
        covar_dim (int, optional): Dimension of the covariate tensor. Defaults to 10.
    """

    def __init__(
        self,
        temporal_PE,
        dim,
        depth,
        heads,
        window_size,
        mlp_dim,
        num_time,
        dropout=0.0,
        device=None,
        covar_dim=10,
    ):
        super().__init__()
        self.temporal_PE = temporal_PE
        if temporal_PE:
            self.pos_embedding = nn.Parameter(torch.randn(num_time, dim))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TemporalAttention(
                            dim=dim,
                            heads=heads,
                            window_size=window_size,
                            dropout=dropout,
                            device=device,
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, covars):
        """
        Forward pass of the TemporalTransformer module.

        Args:
            x (torch.Tensor): Input tensor of shape [b, c, n, t].
            covars (torch.Tensor): Covariate tensor of shape [b, c, 1, t].

        Returns:
            torch.Tensor: Transformed tensor of shape [b, c, n, t].
            torch.Tensor: Covariate tensor of shape [b, c, 1, t].
        """
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)  # [b*n, t, c]
        if self.temporal_PE:
            x = x + self.pos_embedding  # [b*n, t, c]

        covars = covars.permute(0, 2, 3, 1).reshape(b, t, c)  # [b, t, c]
        x = torch.cat([covars, x], dim=0)  # [b+b*n, t, c]

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        covars = x[:b].reshape(b, 1, t, c).permute(0, 3, 1, 2)  # [b, c, 1, t]
        x = x[b:].reshape(b, n, t, c).permute(0, 3, 1, 2)  # [b, c, n, t]
        return x, covars
