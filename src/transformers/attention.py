import torch
from check_shapes import check_shapes
from torch import nn


class SelfAttentionHead(nn.Module):
    """Self Attention Transformer Mechanism."""

    def __init__(self, in_dim: int, out_dim: int, cpu: bool = True) -> None:
        """A custom implementation of a single self-attention head using dot-product
        weighting and softmax activation.

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
