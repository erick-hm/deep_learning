import torch


class TorchDataset(torch.utils.data.dataset.Dataset):
    """Utility class to format raw data into a pytorch dataset, for use with data loaders."""

    def __init__(self, input: torch.Tensor, target: torch.Tensor, transform=None, target_transform=None):
        self.input = input
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.input)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """Retrieve a data point at a particular index."""
        input = self.input[idx]
        target = self.target[idx]

        # optional transforms
        if self.transform:
            input = self.transform(input)

        if self.target_transform:
            target = self.target_transform(target)

        return input, target
