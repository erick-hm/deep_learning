import torch
from check_shapes import check_shapes
from torch import nn


class SelfAttentionHead(nn.Module):
    """Self Attention Transformer Mechanism."""

    def __init__(self, in_dim: int, out_dim: int, cpu: bool = True) -> None:
        """A custom implementation of a single self-attention head using dot-product
        weighting and softmax activation. Designed for batch sizes of 1.

        Params
        ------
        in_dim: int,
            The dimension of the input token vectors (number of features).

        out_dim: int,
            The dimension of the output token vectors (number of new features).

        cpu: bool,
            Whether to use the cpu for calculations. Set to False if a CUDA GPU is
            available.

        Returns
        -------
        None

        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # define the Key, Query and Value matrices
        self.Wk = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")
        self.Wq = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")
        self.Wv = nn.Linear(in_dim, out_dim, bias=True, device="cpu" if cpu else "gpu")

        # apply the softmax across columns // each row must sum to 1
        self.softmax = nn.Softmax(dim=1)

    @check_shapes(
        "X: [batch, in_dim]",
        "return: [batch, out_dim]",
    )
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass calculation.

        Params
        ------
        X: torch.tensor,
            The input token matrix (each row is a token vector).

        Returns
        -------
        out: torch.Tensor,
            The transformed token matrix.

        """
        # calculate the dot product matrices with each token vector
        K = self.Wk(X)
        Q = self.Wq(X)
        V = self.Wv(X)

        # scale down the query-key dot product with growing vector magnitude
        scaled_dot_prod = torch.matmul(Q, K.T) / (self.out_dim**0.5)

        # calculate the linear softmax component
        linear_softmax = self.softmax(scaled_dot_prod)

        # apply activation matrix to the value matrix
        out = torch.matmul(linear_softmax, V)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, in_dim: int, cpu: bool = True):
        """..."""
        super().__init__()

        self.in_dim = in_dim
        self.heads = heads

        # check the out_dim is divisible by number of heads
        assert in_dim % heads == 0
        self.dims_per_head = self.in_dim // heads

        # instantiate the key, query, value and unifying weight matrices
        self.Wk = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")
        self.Wq = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")
        self.Wv = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")
        self.Wunify = nn.Linear(in_dim, in_dim, bias=True, device="cpu" if cpu else "gpu")

        self.softmax = nn.Softmax(2)

    def forward(self, X: torch.Tensor):
        """..."""
        # get the shape of the input tensor (accounting for batch size)
        batch, tokens, features = X.size()

        # calculate the full query, key and values
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wk(X)

        # reshape the tensors per head
        key = K.view(batch, tokens, self.heads, self.dims_per_head)
        query = Q.view(batch, tokens, self.heads, self.dims_per_head)
        value = V.view(batch, tokens, self.heads, self.dims_per_head)

        # incorporate heads into batch dimension
        key = key.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.dims_per_head)
        query = query.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.dims_per_head)
        value = value.transpose(1, 2).contiguous().view(batch * self.heads, tokens, self.dims_per_head)

        # find the query-key dot products using batch matrix multiplication
        scaled_dot_prod = torch.bmm(query, key.transpose(1, 2)) / (self.dims_per_head**0.5)

        # normalize using softmax
        linear_softmax = self.softmax(scaled_dot_prod)

        # get the output for each head by multiplying activations with values
        out = torch.bmm(linear_softmax, value).view(batch, self.heads, tokens, self.dims_per_head)

        # concatenate the heads and apply unifying weight matrix
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.heads * self.dims_per_head)
        out = self.Wunify(out)

        return out
