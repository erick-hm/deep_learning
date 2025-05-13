import torch
from torch.nn import Linear
from torch.utils.data import DataLoader

from losses.vae_loss import VAELoss


class VariationalEncoder(torch.nn.Module):
    """A variational encoder class that maps a data point to a latent vector
    for the mean and the log-variance of the latent representation.

    It implements a configurable:
    - encoder shape and size
    - latent dimension
    - activation function

    Params
    ------
    encoder_shape: tuple[int],
        A tuple indicating the size of each layer of the encoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    hidden_dim: int,
        The dimension of the latent space representation of the data.

    activation: Optional,
        An activation function to use in the network. If None is passed, it defaults to
        a SiLU activation.
    """

    def __init__(self, encoder_shape: tuple[int], hidden_dim: int, activation=None):
        super().__init__()

        # define the encoder layers
        self.encoder_shape = encoder_shape
        self.hidden_dim = hidden_dim
        self.layers = torch.nn.ModuleList()
        self.mean_layer = torch.nn.Linear(encoder_shape[-1], hidden_dim)
        self.log_var_layer = torch.nn.Linear(encoder_shape[-1], hidden_dim)

        # optionally override the activation function
        self.activation = activation
        if activation is None:
            self.activation = torch.nn.SiLU()

        # loop through encoder shape and add layers to the network
        for i in range(len(self.encoder_shape) - 1):
            self.layers.append(Linear(self.encoder_shape[i], self.encoder_shape[i + 1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate a forward pass of the network."""
        for layer in self.layers:
            X = layer(X)
            X = self.activation(X)

        # calculate the mean and log variance from the final layer
        mean = self.mean_layer(X)
        log_var = self.log_var_layer(X)

        return mean, log_var


class VariationalDecoder(torch.nn.Module):
    """A variational decoder class that maps a latent vector to a data point.

    It implements a configurable:
    - decoder shape and size
    - latent dimension
    - activation function

    Params
    ------
    decoder_shape: tuple[int],
        A tuple indicating the size of each layer of the decoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    hidden_dim: int,
        The dimension of the latent space representation of the data.

    activation: Optional,
        An activation function to use in the network. If None is passed, it defaults to
        a SiLU activation.
    """

    def __init__(self, decoder_shape: tuple[int], hidden_dim: int, activation=None):
        super().__init__()

        # define the network shape
        self.decoder_shape = (hidden_dim, *decoder_shape)
        self.hidden_dim = hidden_dim

        # apply the optional activation functions
        self.activation = activation
        if activation is None:
            self.activation = torch.nn.SiLU()

        # construct the layers of the network
        self.layers = torch.nn.ModuleList()
        for i in range(len(self.decoder_shape) - 1):
            self.layers.append(Linear(self.decoder_shape[i], self.decoder_shape[i + 1]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Calculate a forward pass of the network"""
        X_hat = z
        for idx, layer in enumerate(self.layers):
            X_hat = layer(X_hat)
            # only apply activations for pre-final layers
            if idx < len(self.layers) - 1:
                X_hat = self.activation(X_hat)

        return X_hat


class VAE(torch.nn.Module):
    """A variational autoencoder class that learns a low dimensional representation
    of the original data inputs and maps it to a standard normal prior.

    It implements a configurable:
    - encoder/decoder shape and size
    - latent dimension
    - activation function

    Params
    ------
    encoder_shape: tuple[int],
        A tuple indicating the size of each layer of the encoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    decoder_shape: tuple[int],
        A tuple indicating the size of each layer of the decoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    hidden_dim: int,
        The dimension of the latent space representation of the data.

    activation: Optional,
        An activation function to use in the network. If None is passed, it defaults to
        a SiLU activation.
    """

    def __init__(self, encoder_shape: tuple[int], decoder_shape: tuple[int], hidden_dim: int, activation=None):
        super().__init__()
        self.encoder = VariationalEncoder(encoder_shape, hidden_dim, activation=activation)
        self.decoder = VariationalDecoder(decoder_shape, hidden_dim, activation=activation)
        self.hidden_dim = hidden_dim

    def reparametrisation(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparametrise the latent vector using a standard normal
        random variable epsilon.

        Params
        ------
        mean: torch.Tensor,
            The mean predicted by the encoder network.

        log_var: torch.Tensor,
            The log-variance predicted by the encoder network.

        Returns
        -------
        z: torch.Tensor

        """
        # convert from log variance to standard deviation
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + (std * epsilon)

        return z

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate a forward pass of the network.

        Returns
        -------
        X_hat: torch.Tensor,
            The reconstructed data points from the latent representation.

        mean: torch.Tensor,
            The means of the latents.

        log_var: torch.Tensor,
            The log-variance of the latents.

        """
        mean, log_var = self.encoder(X)
        z = self.reparametrisation(mean, log_var)
        X_hat = self.decoder(z)

        return X_hat, mean, log_var


class VAEModel(torch.nn.Module):
    """A wrapper class for the VAE class that implements:
    - fit method, to fit to training data
    - predict methods, to predict on new data
    - a method to generate new data samples

    A variational autoencoder class that learns a low dimensional representation
    of the original data inputs and maps it to a standard normal prior.

    It implements a configurable:
    - encoder/decoder shape and size
    - latent dimension
    - activation function

    Params
    ------
    encoder_shape: tuple[int],
        A tuple indicating the size of each layer of the encoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    decoder_shape: tuple[int],
        A tuple indicating the size of each layer of the decoder network.
        Both the length of the tuple and the value of each entry affect
        the network shape.

    hidden_dim: int,
        The dimension of the latent space representation of the data.

    activation: Optional,
        An activation function to use in the network. If None is passed, it defaults to
        a SiLU activation.
    """

    def __init__(self, encoder_shape: tuple[int], decoder_shape: tuple[int], hidden_dim: int, activation=None) -> None:
        super().__init__()
        self.model = VAE(encoder_shape, decoder_shape, hidden_dim, activation)

    def fit(
        self,
        loss: VAELoss,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Fit the model to training data.

        Params
        ------
        loss: VAELoss,
            The loss function to use.

        optimiser: torch.optim.Optimizer,
            The optimizer to use for weights tuning.

        epochs: int,
            The number of epochs to train for.

        train_loader: DataLoader,
            The data to train on.

        val_loader: DataLoader,
            The validation data to validate against during the training process.

        Returns
        -------
        None

        """

        def validation_loss(val_loader, loss_fn):
            running_vloss = 0
            self.model.eval()
            # disable gradients
            with torch.no_grad():
                for _idx, vdata in enumerate(val_loader):
                    input, target = vdata

                    X_hat, mean, log_var = self.model(input)

                    loss_ = loss_fn(X_hat, target, mean, log_var)
                    running_vloss += loss_.item()

            return running_vloss

        # loop over epochs
        for epoch in range(epochs):
            self.model.train(True)
            running_loss = 0

            # loop over training samples
            for _idx, data in enumerate(train_loader):
                input, target = data

                # zero the gradients
                optimizer.zero_grad()

                # predict
                X_hat, mean, log_var = self.model(input)

                # calculate loss and gradients
                loss_ = loss(X_hat, target, mean, log_var)
                loss_.backward()

                # adjust weights
                optimizer.step()

                # training data
                running_loss += loss_.item()

            # update the annealing procedure for the KL loss
            loss.step_epochs()

            # calculate the validation loss
            val_loss = validation_loss(val_loader, loss)

            print(f"Epoch {epoch} training loss: {round(running_loss, 4)} and validation loss: {round(val_loss, 4)}")
            print()

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Predict the output of new data.

        Params
        ------
        input: torch.Tensor,
            The input data to predict.

        Returns
        -------
        torch.Tensor,
            The reconstructed data.

        """
        self.model.eval()
        return self.model(input)

    def predict_dataset(self, input: DataLoader) -> torch.Tensor:
        """Predict the output of new data stored in a data loader.

        Params
        ------
        input: DataLoader,
            The input data to predict.

        Returns
        -------
        torch.Tensor,
            The reconstructed data.

        """
        self.model.eval()
        all_reconstructions = []
        all_means = []
        all_log_vars = []

        with torch.no_grad():
            for batch in input:
                # ignoring targets
                X, _ = batch
                X_hat, mean, log_var = self.model(X)

                all_reconstructions.append(X_hat)
                all_means.append(mean)
                all_log_vars.append(log_var)

        X_hat = torch.cat(all_reconstructions, dim=0)
        mean = torch.cat(all_means, dim=0)
        log_var = torch.cat(all_log_vars, dim=0)

        return X_hat, mean, log_var

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Sample latent vectors from the standard normal prior.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        Returns
        -------
        torch.Tensor,
            Samples generated.

        """
        return torch.randn(num_samples, self.model.hidden_dim)

    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """Generate synthetic data samples by sampling from the standard
        normal prior and reconstructing the latent representation to a
        data point.

        Params
        ------
        num_samples: int,
            The number of samples to generate.

        Returns
        -------
        torch.Tensor,
            The generated synthetic data.

        """
        sampled_latent = self.sample_prior(num_samples)

        return self.model.decoder(sampled_latent)
