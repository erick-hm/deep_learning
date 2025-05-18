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
    kl_weight: float = 1.0,
        Sets the relative importance of the KL-divergence compared to the
        reconstruction error.

    warmup_epochs: int = 10,
        Sets the rate of annealing of the KL-divergence. Higher values indicate
        a slower ramp-up of the KL divergence contribution to the loss.

    reduction: Literal['mean','sum','none'],
        The method for aggregating the loss.
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        warmup_epochs: int = 10,
        reduction: Literal["mean", "sum", "none"] = "mean",
        loss_type: Literal["mse", "l1"] = "mse",
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.warmup_epochs = warmup_epochs  # added for annealing of the KL loss term
        self.current_epoch = 0

        if loss_type not in ("mse", "l1"):
            raise ValueError("Loss type must be 'mse', 'l1'")
        self.loss_type = loss_type

        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.reduction = reduction

    def forward(self, X_hat: torch.Tensor, X: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
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
        """
        if self.loss_type == "mse":
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(X_hat, X, reduction="none")
        if self.loss_type == "l1":
            recon_loss = F.l1_loss(X_hat, X, reduction="none")

        recon_loss = recon_loss.sum(dim=1)  # sum over features per sample

        # KL divergence loss assuming multivariate Gaussian prior and posterior
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

        # Combine the loss
        kl_weight = self.kl_weight * min(self.current_epoch / self.warmup_epochs, 1.0)

        loss = recon_loss + (kl_weight * kl_div)

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
