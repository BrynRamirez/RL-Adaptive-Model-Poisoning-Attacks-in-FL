from torchvision import datasets, transforms
import torch
import numpy as np
import random

def get_non_iid_emnist_dataloader(num_clients=10, batch_size=32, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)

    train_labels = np.array(train_dataset.targets)
    test_labels = np.array(test_dataset.targets)
    num_classes = len(set(train_labels))

    # Non-IID split by classes
    shards_per_client = num_classes // num_clients
    class_indices = np.arange(num_classes)
    np.random.shuffle(class_indices)

    client_classes = [class_indices[i * shards_per_client:(i + 1) * shards_per_client] for i in range(num_clients)]

    def get_client_subset(dataset, labels, client_class_set):
        indices = [i for i, label in enumerate(labels) if label in client_class_set]
        return torch.utils.data.Subset(dataset, indices)

    train_subsets = [get_client_subset(train_dataset, train_labels, client_classes[i]) for i in range(num_clients)]
    test_subsets = [get_client_subset(test_dataset, test_labels, client_classes[i]) for i in range(num_clients)]

    train_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
                     for subset in train_subsets]
    test_loaders = [torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
                    for subset in test_subsets]

    # Match malicious clients with cid % 5 == 0
    malicious_indices = [cid for cid in range(num_clients) if cid % 5 == 0]
    clean_indices = [cid for cid in range(num_clients) if cid % 5 != 0]

    clean_test_data = torch.utils.data.ConcatDataset([test_subsets[i] for i in clean_indices])
    malicious_test_data = torch.utils.data.ConcatDataset([test_subsets[i] for i in malicious_indices])

    clean_test_loader = torch.utils.data.DataLoader(clean_test_data, batch_size=batch_size, shuffle=False)
    malicious_test_loader = torch.utils.data.DataLoader(malicious_test_data, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loaders, clean_test_loader, malicious_test_loader
