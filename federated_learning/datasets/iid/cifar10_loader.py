import torch
from torchvision import datasets, transforms
import numpy as np
import random

def get_iid_cifar10_dataloader(num_clients=10, batch_size=32, seed=42):
    """Loads and partitions CIFAR10 dataset into IID subsets for federated learning clients."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Compute the number of samples per client
    train_samples_per_client = len(train_dataset) // num_clients
    test_samples_per_client = len(test_dataset) // num_clients

    def partition_dataset(dataset, samples_per_client):
        """Helper function to partition dataset into client subsets."""
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

    train_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in
                         train_subsets]
    test_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False) for subset in
                        test_subsets]

    # Central loaders (shared evaluation)
    test_clean_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_malicious_loader = test_clean_loader  # Placeholder for now

    return train_loaders, test_loaders, test_clean_loader, test_malicious_loader

