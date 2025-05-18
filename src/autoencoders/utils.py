from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from autoencoders.vae import VAEModel


def diagnose_posterior_collapse(
    vae: VAEModel,
    data_loader: DataLoader,
    device: Literal["cpu", "gpu"] = "cpu",
    latent_dim: int = 1,
    num_samples: int = 500,
):
    """Show the posterior distribution of the mean and log-variance for different
    input samples. It plots the histogram of means and shows the how well the
    synthetic data reconstructed from points sampled from a standard normal
    distribution compares to those sampled from the posterior.

    Params
    ------
    vae: VAEModel,
        A VAE with an encoder and decoder attribute.

    data_loader: DataLoader,
        A data loader with real data samples to sample from the posterior.

    device: Literal['cpu', 'gpu'],
        Which device to move the data to.

    latent_dim: int,
        The dimension of the latent space.

    num_samples: int,
        How many samples to use for plotting purposes.
    """
    vae.eval()
    mu_list = []
    std_list = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            x = batch[0].to(device)
            mu, log_var = vae.model.encoder(x)
            mu_list.append(mu)
            std_list.append(torch.exp(0.5 * log_var))
            if i * x.size(0) >= num_samples:
                break

    mu_all = torch.cat(mu_list, dim=0)
    std_all = torch.cat(std_list, dim=0)

    print("Mean of latent means:", mu_all.mean(dim=0))
    print("Std of latent means:", mu_all.std(dim=0))
    print("Mean of latent stds:", std_all.mean(dim=0))
    print("Std of latent stds:", std_all.std(dim=0))

    # Plot the latent mean distribution (1D or 2D)
    if latent_dim == 1:
        plt.figure(figsize=(6, 4))
        plt.hist(mu_all.cpu().flatten().tolist(), bins=40, alpha=0.7)
        plt.title("Histogram of Latent Means (mu)")
        plt.xlabel("z")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()
    elif latent_dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(mu_all[:, 0].cpu().squeeze().tolist(), mu_all[:, 1].cpu().squeeze().tolist(), alpha=0.5)
        plt.title("Latent Mean Distribution")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.axis("equal")
        plt.grid(True)
        plt.show()
    else:
        print("Only configured to plot for 1D and 2D latent spaces.")

    # Evaluate reconstruction from encoded z and random z
    x = data_loader.dataset.input.to(device)[:num_samples]
    mu, log_var = vae.model.encoder(x)
    z_encoded = vae.model.reparametrisation(mu, log_var)

    # sample |z_encoded| samples from a standard normal
    z_random = torch.randn_like(z_encoded)

    # reconstruct from the data and from the latent prior
    x_hat_encoded = vae.model.decoder(z_encoded)
    x_hat_random = vae.model.decoder(z_random)

    fig, axes = plt.subplots(3, 1, figsize=(6, 12))
    for _idx, (ax, data, title) in enumerate(
        zip(
            axes,
            [x.cpu(), x_hat_encoded.cpu(), x_hat_random.cpu()],
            ["Original Input", "Reconstruction (Encoded z)", "Reconstruction (Random z)"],
            strict=False,
        )
    ):
        ax.scatter(data[:, 0].tolist(), data[:, 1].tolist())
        ax.set_title(title)
        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize="x-small")
        ax.grid(True)

    plt.xlabel("Feature Dimension")
    plt.tight_layout()
    plt.show()
