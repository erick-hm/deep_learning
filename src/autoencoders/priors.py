from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, model_validator

from src.exceptions import ShapeMismatchError
from src.schema import PositiveFloat


class Prior(ABC, BaseModel):
    """An abstract base class for priors to be used in VAEs."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the prior distribution."""
        pass


class GaussianPrior(Prior):
    """A Gaussian distribution for use as a prior.

    Params
    ------
    mean: float,
        The mean of the Gaussian distribution.

    sigma: PositiveFloat,
        The standard deviation of the Gaussian distribution.
    """

    mean: float
    sigma: PositiveFloat
    as_torch_params: bool = False

    def sample(self, num_samples: int, latent_dim: int) -> torch.Tensor:
        """Generate samples from a Gaussian distribution with specified
        mean and standard deviation.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        latent_dim: int,
            The dimension of the latent space.

        Returns
        -------
        torch.Tensor,
            A pytorch tensor containing the samples.

        """
        samples = torch.normal(self.mean, self.sigma, (num_samples, latent_dim))

        return samples

    @model_validator(mode="after")
    def validate_torch_params(self):
        """Conditionally convert class parameters to Pytorch parameters."""
        if self.as_torch_params:
            self.mean = torch.nn.Parameter(torch.Tensor(self.mean))
            self.sigma = torch.nn.Parameter(torch.Tensor(self.sigma))

        return self


class GaussianMixturePrior(Prior):
    """A Gaussian Mixture distribution for use as a prior.

    Params
    ------
    weights: list[PositiveFloat],
        The weights of the distinct Gaussian distributions.

    means: list[float],
        The means of the distinct Gaussian distributions.

    sigma: PositiveFloat,
        The standard deviations of the distinct Gaussian distributions.
    """

    weights: list[PositiveFloat]
    means: list[float]
    sigmas: list[PositiveFloat]
    as_torch_params: bool = False

    def __len__(self):
        return len(self.weights)

    def sample(self, num_samples: int, latent_dim: int) -> torch.Tensor:
        """Generate samples from a Gaussian mixture distribution with specified
        mean and standard deviation.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        latent_dim: int,
            The dimension of the latent space.

        Returns
        -------
        torch.Tensor,
            A pytorch tensor containing the samples.

        """
        weights_sample_idxs = torch.multinomial(torch.tensor(self.weights), num_samples=num_samples, replacement=True)
        sample_means = torch.tensor(self.means)[weights_sample_idxs]
        sample_sigmas = torch.tensor(self.sigmas)[weights_sample_idxs]

        samples = torch.normal(sample_means, sample_sigmas, (num_samples, latent_dim))

        return samples

    @model_validator(mode="after")
    def validate_shapes(self) -> None:
        """Validate that weights, means and sigmas are the same shape."""
        length = len(self.weights)

        if (len(self.means) != length) or (len(self.sigmas) != length):
            msg = "Weights, means and sigmas should all have the same shape."
            raise ShapeMismatchError(msg)

        return self

    @model_validator(mode="after")
    def validate_weights(self):
        """Validate that the weights sum to one."""
        weights_sum = round(sum(self.weights), 3)
        if weights_sum != 1:
            msg = "Weights array must sum to one."
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def validate_torch_params(self):
        """Conditionally convert class parameters to Pytorch parameters."""
        if self.as_torch_params:
            self.weights = torch.nn.Parameter(torch.Tensor(self.weights))
            self.means = torch.nn.Parameter(torch.Tensor(self.means))
            self.sigmas = torch.nn.Parameter(torch.Tensor(self.sigmas))

        return self
