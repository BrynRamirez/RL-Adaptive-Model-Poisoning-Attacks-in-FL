import torch
from torchvision import datasets, transforms
import numpy as np
import random


def get_iid_mnist_dataloader(num_clients=10, batch_size=32, seed=42):
    """Loads and partitions MNIST dataset into IID subsets for federated learning clients."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Partitioning helper
    def partition_dataset(dataset, samples_per_client):
        subsets = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            subset = torch.utils.data.Subset(dataset, list(range(start_idx, end_idx)))
            subsets.append(subset)
        return subsets

    # Per-client split
    train_samples_per_client = len(train_dataset) // num_clients
    test_samples_per_client = len(test_dataset) // num_clients
    train_subsets = partition_dataset(train_dataset, train_samples_per_client)
    test_subsets = partition_dataset(test_dataset, test_samples_per_client)

    train_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in train_subsets]
    test_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False) for subset in test_subsets]

    # Central loaders (shared evaluation)
    test_clean_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_malicious_loader = test_clean_loader  # Placeholder for now

    return train_loaders, test_loaders, test_clean_loader, test_malicious_loader

