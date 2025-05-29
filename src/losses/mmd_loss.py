from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from check_shapes import check_shapes


@check_shapes("z1: [N,features]", "z2: [M,features]", "return: [N, M]")
def gaussian_kernel(z1: torch.Tensor, z2: torch.Tensor, sigma: float = 1) -> torch.Tensor:
    """Compute the pairwise Gaussian kernel for a batch of latent vectors.

    Params
    ------
    z1: torch.Tensor,
        The first batch of latent vectors. Tensor of shape [N, features].

    z2: torch.Tensor,
        The second batch of latent vectors. Tensor of shape [M, features].

    sigma: float,
        The scaling factor, equivalent to the standard deviation in a Gaussian.
        Must be greater than 0.

    Returns
    -------
    torch.Tensor,
        The pairwise Gaussian kernel matrix of shape [N, M], where each
        element (i,j) contains exp(-||z1_i - z2_j||^2 / (2*sigma^2)).

    """
    assert sigma > 0, "Sigma must be a positive valued float."

    # reshape to [N, 1, features] and [1, M, features]
    z1 = check_shapes(z1.unsqueeze(1), "[N, 1, features]")
    z2 = check_shapes(z2.unsqueeze(0), "[1, M, features]")

    # calculate the square difference and sum over the feature dimension
    diff = check_shapes((z1 - z2), "[N, M, features]")

    # sum over the feature dimension
    sq_dist = diff.pow(2).sum(dim=2)
    scale = 2 * (sigma**2)

    return torch.exp(-sq_dist / scale)


class MMDLoss(torch.nn.Module):
    """Implements a (biased) Maximum Mean Discrepancy loss for MMD-VAE.
    Using a modifiable kernel function k(z, z'), it computes the MMD as:

    MMD = E_prior[k(z_prior, z_prior')] + E_posterior[k(z_posterior, z_posterior')]
            - 2 E_prior_posterior[k(z_prior, z_posterior)]

    The overall loss is given as:

    Loss = (Reconstruction Loss) + alpha (MMD Loss)

    Params
    ------
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        A function of two input tensors corresponding to latent samples.

    alpha: float,
        The relative strength of the MMD loss versus the reconstruction loss.
    """

    def __init__(
        self,
        kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        alpha: float = 1,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()

        self.kernel_fn = kernel_fn
        self.alpha = float(alpha)

        if reduction not in ("mean", "sum"):
            msg = "Reduction parameter must be one of 'mean' or 'sum'."
            raise ValueError(msg)
        self.reduction = reduction

    @check_shapes(
        "z_prior: [batch, features]",
        "z_posterior: [batch, features]",
        "X: [batch, in_features]",
        "X_hat: [batch, in_features]",
        "return: []",
    )
    def forward(
        self, z_prior: torch.Tensor, z_posterior: torch.Tensor, X: torch.Tensor, X_hat: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss for the provided samples.

        Params
        ------
        z_prior: torch.Tensor,
            Samples of the latent vector drawn from a prior distribution.

        z_posterior: torch.Tensor,
            Samples of the posterior latent vector for real data inputs.

        X: torch.Tensor,
            The input data.

        X_hat: torch.Tensor,
            The reconstructed input data.

        Returns
        -------
        loss: torch.Tensor,
            A scalar indicating the global loss for the batched sample.

        """
        prior_prior = self.kernel_fn(z_prior, z_prior)
        prior_posterior = self.kernel_fn(z_prior, z_posterior)
        posterior_posterior = self.kernel_fn(z_posterior, z_posterior)

        mmd_loss = prior_prior.mean() + posterior_posterior.mean() - (2 * prior_posterior.mean())

        reconstruction_loss = F.mse_loss(X, X_hat, reduction=self.reduction)

        loss = reconstruction_loss + (self.alpha * mmd_loss)

        return loss
