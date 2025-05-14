import torch
from pydantic import BaseModel, Field


class CircleDataGenerator(BaseModel):
    """Create samples from a circle with specified radius, centre and noise.

    Params
    ------
    radius: float,
        The radius of the circle. Must be non-negative.

    x_centre: float,
        The centre of the circle in the x dimension.

    y_centre: float,
        The centre of the circle in the y dimension.

    std_dev: float,
        The standard deviation to use for the noise for a 0-centred
        normal distribution.
    """

    radius: float = Field(default=1, ge=0)
    x_centre: float = Field(default=0)
    y_centre: float = Field(default=0)
    std_dev: float = Field(default=0.01, ge=0)

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate data samples drawn from a noisy circle distribution.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        Returns
        -------
        samples: torch.Tensor,
            The samples as a 2D tensor.

        """
        # sample theta from a uniform distribution and noise from a normal distribution
        theta = torch.rand(num_samples) * 2 * torch.pi
        noise = torch.distributions.Normal(loc=0.0, scale=self.std_dev)

        # sample the data points in the 2D plane
        X = (self.radius * torch.cos(theta)) + noise.sample((num_samples,)) + self.x_centre
        Y = (self.radius * torch.sin(theta)) + noise.sample((num_samples,)) + self.y_centre

        # concatenate the data into one 2D tensor
        samples = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        return samples


class SemiCircleDataGenerator(CircleDataGenerator):
    """Create samples from a semi-circle with specified radius, centre and noise.

    Params
    ------
    radius: float,
        The radius of the circle. Must be non-negative.

    x_centre: float,
        The centre of the circle in the x dimension.

    y_centre: float,
        The centre of the circle in the y dimension.

    std_dev: float,
        The standard deviation to use for the noise for a 0-centred
        normal distribution.
    """

    def _rotate_data(self, data: torch.Tensor, angle: float) -> torch.Tensor:
        angle = torch.Tensor([float(angle)])
        # convert angle to radians
        angle = torch.pi * angle / 180

        # rotate the 2d data
        rotation_matrix = torch.Tensor([[torch.cos(angle), torch.sin(angle)], [-torch.sin(angle), torch.cos(angle)]])
        rotated_data = data @ rotation_matrix

        return rotated_data

    def generate_samples(self, num_samples: int, angle: float = 0) -> torch.Tensor:
        """Generate data samples drawn from a noisy semi-circle distribution.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        angle: float,
            The angle in degrees to rotate the semi-circle about anti-clockwise.

        Returns
        -------
        samples: torch.Tensor,
            The samples as a 2D tensor.

        """
        # sample theta from a uniform distribution and noise from a normal distribution
        theta = torch.rand(num_samples) * 2 * torch.pi
        noise = torch.distributions.Normal(loc=0.0, scale=self.std_dev)

        # sample the data points in the 2D plane
        X = (self.radius * torch.cos(theta)) + noise.sample((num_samples,)) + self.x_centre
        # take the absolute value to make it a semi-circle
        Y = torch.abs((self.radius * torch.sin(theta)) + noise.sample((num_samples,)) + self.y_centre)

        # concatenate the data into one 2D tensor
        samples = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

        # rotate the data to the desired angle
        samples = self._rotate_data(samples, angle)

        return samples
