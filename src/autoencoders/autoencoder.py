import torch
from torch.nn import L1Loss, Linear, MSELoss
from torch.utils.data import DataLoader


class Encoder(torch.nn.Module):
    """An encoder class that maps a data point to a lower dimensional
    latent vector.

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

    def __init__(self, encoder_shape: tuple[int], hidden_dim: int, activation=None) -> None:
        super().__init__()
        self.encoder_shape = (*encoder_shape, hidden_dim)
        self.hidden_dim = hidden_dim

        # optionally override the activation function
        self.activation = activation
        if activation is None:
            self.activation = torch.nn.SiLU()

        # loop through encoder shape and add layers to the network
        self.layers = torch.nn.ModuleList()
        for i in range(len(self.encoder_shape) - 1):
            self.layers.append(Linear(self.encoder_shape[i], self.encoder_shape[i + 1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate a forward pass of the network."""
        for layer in self.layers:
            X = layer(X)
            X = self.activation(X)

        return X


class Decoder(torch.nn.Module):
    """A decoder class that maps a latent vector to a data point.

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

    def __init__(self, decoder_shape: tuple[int], hidden_dim: int, activation=None) -> None:
        super().__init__()
        self.decoder_shape = (hidden_dim, *decoder_shape)
        self.hidden_dim = hidden_dim

        # optionally override the activation function
        self.activation = activation
        if activation is None:
            self.activation = torch.nn.SiLU()

        # loop through encoder shape and add layers to the network
        self.layers = torch.nn.ModuleList()
        for i in range(len(self.decoder_shape) - 1):
            self.layers.append(Linear(self.decoder_shape[i], self.decoder_shape[i + 1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate a forward pass of the network."""
        for layer in self.layers:
            X = layer(X)
            X = self.activation(X)

        return X


class AE(torch.nn.Module):
    """An autoencoder class that maps a data point to a latent vector and back
    to a data point.

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
        self.encoder = Encoder(encoder_shape, hidden_dim, activation=activation)
        self.decoder = Decoder(decoder_shape, hidden_dim, activation=activation)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate a forward pass of the network."""
        X = self.encoder(X)
        X = self.decoder(X)

        return X


class AEModel(torch.nn.Module):
    """A wrapper class for the autoencoder model. It has inbuilt fit and predict
    methods to simplify the training and inference processes.

    An autoencoder class that maps a data point to a latent vector and back
    to a data point.

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
        self.model = AE(encoder_shape, decoder_shape, hidden_dim, activation)

    def fit(
        self,
        loss: MSELoss | L1Loss,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        """Fit the model to training data.

        Params
        ------
        loss: MSELoss | L1Loss,
            The loss function to use.

        optimiser: torch.optim.Optimizer,
            The optimizer to use for weights tuning.

        epochs: int,
            The number of epochs to train for.

        train_loader: DataLoader,
            The data to train on.

        val_loader: DataLoader | None,
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
                    output = self.model(input)
                    loss_ = loss_fn(output, target)
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
                output = self.model(input)

                # calculate loss and gradients
                loss_ = loss(output, target)
                loss_.backward()

                # adjust weights
                optimizer.step()

                # training data
                running_loss += loss_.item()

            # calculate the validation loss
            val_loss = validation_loss(val_loader, loss)

            print(f"Epoch {epoch} training loss: {round(running_loss, 4)} and validation loss: {round(val_loss, 4)}")

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Run inference on new data points.

        Params
        ------
        input: torch.Tensor,
            The new data to run inference on.

        Returns
        -------
        torch.Tensor,
            The reconstructed data.

        """
        self.model.eval()
        return self.model(input)

    def predict_dataset(self, input: DataLoader):
        """Run inference on a pytorch dataset stored in a data loader."""
        msg = "To be done"
        raise NotImplementedError(msg)
