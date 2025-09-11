# Load Dataset Function
def load_dataset(dataset_type="iid", dataset_name="mnist", num_clients=20, batch_size=32, seed=42):
    if dataset_type == "iid" and dataset_name == "mnist":
        from federated_learning.datasets.iid.mnist_loader import get_iid_mnist_dataloader
        return get_iid_mnist_dataloader(num_clients, batch_size, seed)
    elif dataset_type == "iid" and dataset_name == "cifar10":
        from federated_learning.datasets.iid.cifar10_loader import get_iid_cifar10_dataloader
        return get_iid_cifar10_dataloader(num_clients, batch_size, seed)
    elif dataset_type == "non_iid" and dataset_name == 'emnist':
        from federated_learning.datasets.non_iid.emnist_loader import get_non_iid_emnist_dataloader
        return get_non_iid_emnist_dataloader(num_clients, batch_size, seed)
    elif dataset_type == 'non_iid' and dataset_name == "cifar100":
            from federated_learning.datasets.non_iid.cifar100_loader import get_non_iid_cifar100_dataloader
            return get_non_iid_cifar100_dataloader(num_clients, batch_size, seed)
    else:
        raise ValueError("Unknown dataset type or name")
