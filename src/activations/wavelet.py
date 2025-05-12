import torch


class WaveletActivation(torch.nn.Module):
    """Custom activation function with a Gaussian-enveloped sinusoid.

    The activation is localised (approximately zero for both large positive
    and negative inputs), learnable (frequency, decay, magnitude) and highly
    non-linear.

    The high levels of non-linearity allow the activations to be more
    computationally efficient. Meanwhile the locality makes it better for
    interpolation tasks (but poorer for extrapolation).

    Function: slope * sin(alpha * x) * exp(- beta |x|^2)

    Params
    ------
    alpha: float,
        The learnable inverse frequency (or wavenumber) of the wavelet.

    beta: float,
        The learnable strength of the Gaussian decay.

    slope: float,
        The learnable magnitude/amplitude of the wavelet.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, slope: float = 1.0):
        super().__init__()

        # learnable parameters
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float32))
        self.slope = torch.nn.Parameter(torch.Tensor([slope]))

    def forward(self, x: torch.Tensor):
        """Forward pass of the network."""
        return self.slope * torch.sin(self.alpha * x) * torch.exp(-self.beta * (x**2))


class DampedWaveletActivation(torch.nn.Module):
    """Custom activation function with a Gaussian-enveloped sinusoid with modulated
    shape. This is a variation of the WaveletActivation.

    The activation is localised (approximately zero for both large positive
    and negative inputs), learnable (frequency, decay, magnitude) and highly
    non-linear.

    The high levels of non-linearity allow the activations to be more
    computationally efficient. Meanwhile the locality makes it better for
    interpolation tasks (but poorer for extrapolation).

    Function: slope * sign(sin(alpha * x)) * |sin(alpha * x)|^gamma * exp(- beta |x|^2)
    Params
    ------
    alpha: float,
        The learnable inverse frequency (or wavenumber) of the wavelet.

    beta: float,
        The learnable strength of the Gaussian decay.

    slope: float,
        The learnable magnitude/amplitude of the wavelet.

    gamma: float,
        The learnable modulation of the wave. It raises the sinusoid to a power
        to vary the shape of the wave to create flatter or sharper peaks and troughs.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, slope: float = 1.0, gamma: float = 1.0):
        super().__init__()

        # learnable parameters
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.float32))
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.float32))
        self.gamma = torch.nn.Parameter(torch.tensor([gamma], dtype=torch.float32))
        self.slope = torch.nn.Parameter(torch.Tensor([slope]))

    def forward(self, x: torch.Tensor):
        """Forward pass of the network."""
        sin = torch.sin(self.alpha * x)
        # clamp for numerical stability when raising to a power
        abs_sin = torch.clamp(torch.abs(sin), 1e-6, 1)
        sign_sin = torch.where(sin >= 0, 1, -1)

        return self.slope * sign_sin * (abs_sin**self.gamma) * torch.exp(-self.beta * (x**2))
