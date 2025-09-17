import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from federated_learning.datasets import load_dataset
from federated_learning.nn_models.model_loader import load_model

class AdaptiveClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset_type: str, dataset_name: str, seed: int = 42):
        self.cid = int(cid)  # Store client ID
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset dynamically
        train_loader, test_loader, test_clean_loader, test_malicious_loader = load_dataset(self.dataset_type,
                                                                                           self.dataset_name, seed= seed)

        self.train_loader = train_loader[self.cid]
        self.test_loader = test_loader[self.cid]
        self.test_benign_loader = test_clean_loader
        self.test_malicious_loader = test_malicious_loader

        # Define model, loss, and optimizer
        self.model = load_model(dataset_name).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)


    def get_parameters(self, config):
        """Return model parameters."""
        return [param.detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters."""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype)

    def compute_update_norm(self, old_params, new_params):
        old_flat = np.concatenate([p.flatten() for p in old_params])
        new_params = np.concatenate([p.flatten() for p in new_params])
        return np.linalg.norm(new_params - old_flat)

    def train_model(self, num_epochs=1):
        """Train the model and store results."""
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate average loss and accuracy for this epoch
            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracy = 100 * correct / total

    def evaluate_model(self):
        # Already has self.test_loader (combined), self.test_benign_loader, self.test_malicious_loader
        def eval_loader(loader):
            correct, total, loss_sum = 0, 0, 0.0
            self.model.eval()
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y).item()
                    loss_sum += loss
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            return loss_sum / total, correct / total

        full_loss, full_acc = eval_loader(self.test_benign_loader)
        clean_loss, clean_acc = eval_loader(self.test_benign_loader)
        mal_loss, mal_acc = eval_loader(self.test_malicious_loader)

        return {
            "central_global_loss": full_loss,
            "central_global_accuracy": full_acc,
            "central_clean_loss": clean_loss,
            "central_clean_accuracy": clean_acc,
            "central_malicious_loss": mal_loss,
            "central_malicious_accuracy": mal_acc,
        }

    def fit(self, parameters, config):
        """Perform training."""
        self.set_parameters(parameters)
        old_params = [p.copy() for p in parameters]

        self.train_model(num_epochs=1)
        new_params = self.get_parameters(config)
        update_norm = self.compute_update_norm(old_params, new_params)

        return new_params, len(self.train_loader.dataset), {'update_norm': float(update_norm)}

    def evaluate(self, parameters, config):
        """Perform evaluation."""
        self.set_parameters(parameters)
        metrics = self.evaluate_model()
        return float(metrics["local_loss"]), int(len(self.test_loader.dataset)), metrics


def client_fn(cid: str, dataset_type: str, dataset_name: str):
    """Factory function for creating client instances."""
    return AdaptiveClient(cid=cid, dataset_type=dataset_type, dataset_name=dataset_name)
