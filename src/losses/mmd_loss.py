import torch
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
        The scaling factor, equivalenet to the standard deviation in a Gaussian.
        Must be greater than 0.

    Returns
    -------
    torch.Tensor,
        The pairwise Gaussian kernel matrix of shape [N, M].

    """
    assert sigma > 0, "Sigma must be a positive valued float."

    # reshape to [batch, 1, features] and [1, batch, features]
    z1 = z1.unsqueeze(1)
    z2 = z2.unsqueeze(0)

    # calculate the square difference and sum over the feature dimension
    diff = check_shapes((z1 - z2), "[N, M, features]")

    # sum over the feature dimension
    sq_dist = diff.pow(2).sum(dim=2)
    scale = 2 * (sigma**2)

    return torch.exp(-sq_dist / scale)
