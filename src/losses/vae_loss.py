import logging
from typing import Literal

import torch
import torch.nn.functional as F


class VAELoss(torch.nn.Module):
    """Implements a VAE loss using MSE/L1 loss and KL-divergence, assuming
    continuous valued input data and a Gaussian prior/posterior.

    Institutes KL-annealing to gradually increase the importance of
    the KL divergence and prevent failing to minimise reconstruction error.

    The KL weight can be set for use as a beta-VAE.

    Params
    ------
    beta: float = 1.0,
        Sets the relative importance of the KL-divergence compared to the
        reconstruction error. The beta from beta-VAE

    l1_lambda: float = 0,
        Apply an L1 loss to the latent vector to encourage sparsity in the
        representation.

    warmup_epochs: int = 10,
        Sets the rate of annealing of the KL-divergence. Higher values indicate
        a slower ramp-up of the KL divergence contribution to the loss.

    reduction: Literal['mean','sum','none'],
        The method for aggregating the loss.

    dim_agg: Literal["mean", "sum"],
        The aggregation function to use across the loss for each output dimension.

    loss_type: Literal["mse", "l1"],
        Whether to use MSE or L1 loss for the reconstruction error term.
    """

    def __init__(
        self,
        beta: float = 1.0,
        l1_lambda: float = 0.0,
        warmup_epochs: int = 10,
        reduction: Literal["mean", "sum", "none"] = "mean",
        dim_agg: Literal["mean", "sum"] = "mean",
        loss_type: Literal["mse", "l1"] = "mse",
    ):
        super().__init__()
        self.beta = beta
        self.warmup_epochs = warmup_epochs  # added for annealing of the KL loss term
        self.current_epoch = 0
        self.l1_lambda = l1_lambda

        if loss_type not in ("mse", "l1"):
            raise ValueError("Loss type must be 'mse', 'l1'")
        self.loss_type = loss_type

        if dim_agg not in ("mean", "sum"):
            raise ValueError("dim_agg must be 'mean' or 'sum'")
        self.dim_agg = dim_agg

        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.reduction = reduction

    def forward(
        self,
        X_hat: torch.Tensor,
        X: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calculate the loss as the reconstruction error plus the kullback-leibler
        divergence (relative entropy).

        Assumes a multivariate Gaussian prior and posterior.

        Params
        ------
        X_hat: torch.Tensor,
            The reconstructed data samples.

        X: torch.Tensor,
            The original data samples.

        mean: torch.Tensor,
            The latent means predicted by the encoder.

        log_var: torch.Tensor,
            The latent log-variances predicted by the encoder.

        z: torch.Tensor | None,
            The latent vector samples. Only pass when using l1_loss.
        """
        latent_l1_loss = 0
        if self.l1_lambda == 0:
            if z is None:
                msg = "Must pass z if using a non-zero L1 lambda term."
                raise ValueError(msg)
        else:
            # calculate the l1 loss
            latent_l1_loss = self.l1_lambda * z.abs().sum(dim=1)

        if self.loss_type == "mse":
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(X_hat, X, reduction="none")
        if self.loss_type == "l1":
            recon_loss = F.l1_loss(X_hat, X, reduction="none")

        if self.dim_agg == "sum":
            # sum over features per sample
            recon_loss = recon_loss.sum(dim=1)
        elif self.dim_agg == "mean":
            recon_loss = recon_loss.mean(dim=1)

        # KL divergence loss assuming multivariate Gaussian prior and posterior
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

        # anneal the KL weighting
        beta = self.beta * min(self.current_epoch / self.warmup_epochs, 1.0)

        # Combine the losses
        loss = recon_loss + (beta * kl_div) + latent_l1_loss

        logging.debug("kl div", round(kl_div.mean().item(), 3))
        logging.debug("recon_loss", round(recon_loss.mean().item(), 3))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss

    def step_epochs(self):
        """Update the epoch count for the KL-divergence annealing."""
        self.current_epoch += 1
