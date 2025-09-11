def load_model(dataset_name):
    if dataset_name == "mnist":
        from federated_learning.nn_models.mnist_nn import MNIST_NN
        return MNIST_NN()
    elif dataset_name == "cifar10":
        from federated_learning.nn_models.cifar10_nn import CIFAR10_NN
        return CIFAR10_NN()
    elif dataset_name == "cifar100":
        from federated_learning.nn_models.cifar100_nn import CIFAR100_NN
        return CIFAR100_NN()
    elif dataset_name == 'emnist':
        from federated_learning.nn_models.emnist_nn import EMNIST_NN
        return EMNIST_NN()
    else:
        raise ValueError(f"No model found for dataset: {dataset_name}")
