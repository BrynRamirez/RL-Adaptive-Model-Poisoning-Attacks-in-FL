import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from federated_learning.attacks.adaptive_model_poisoning import AdaptiveModelPoisoning
from federated_learning.attacks.deep_rl_adaptive_model_poisoning import RLAdaptiveModelPoisoning

from federated_learning.datasets import load_dataset
from federated_learning.nn_models.model_loader import load_model

_global_poisoner = None

class Poisoned_AdaptiveClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset_type: str, dataset_name: str, is_malicious: bool, attack_strategy: str,
                 seed: int = 42, poisoner=None):
        self.cid = int(cid)  # Store client ID
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load dataset dynamically
        train_loader, test_loader, test_clean_loader, test_malicious_loader = load_dataset(self.dataset_type,
                                                                            self.dataset_name, seed=seed)

        self.train_loader = train_loader[int(self.cid)]
        self.test_loader = test_loader[int(self.cid)]
        self.test_benign_loader = test_clean_loader  # shared across all clients
        self.test_malicious_loader = test_malicious_loader

        self.is_malicious = is_malicious
        self.attack_strategy = attack_strategy
        if self.is_malicious:
            if attack_strategy == 'scale':
                self.poisoner = AdaptiveModelPoisoning(strategy='scale', max_scale=0.5)
            elif attack_strategy == 'noise':
                self.poisoner = AdaptiveModelPoisoning(strategy='noise', noise_lower=0.001, noise_upper=0.1)
            elif attack_strategy == 'rl':
                self.poisoner = poisoner or RLAdaptiveModelPoisoning()
            elif attack_strategy == 'none':
                self.poisoner = None
            else:
                raise ValueError(f'Unknown attack strategy: {attack_strategy}')

        # Define model, loss, and optimizer
        self.model = load_model(dataset_name)

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
                images, labels = images.to(self.device), labels.to(self.device)
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
        """Evaluate the model."""
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_test_loss = loss_total / len(self.test_loader)

        return avg_test_loss, accuracy

    def evaluate_on_loader(self, model, loader):
        model.eval()
        correct, total, loss_total = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_total += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return loss_total / total, correct / total

    def fit(self, parameters, config):
        """Trains local model using local data. In order to apply poisoning, we check if it is a malicious client, has
        a poisoner defined, and if the current round is greater than 10 (allowing the global model to converge)."""
        print(f"[Client {self.cid}] Entering fit. Malicious: {self.is_malicious}")
        self.set_parameters(parameters)
        current_round = config.get("round_num")
        total_rounds = 50
        print(f'[SERVER ROUND: {current_round}]')
        old_params = [p.copy() for p in parameters]

        self.train_model(num_epochs=1)
        if self.is_malicious and self.poisoner is not None and current_round > 10:
            global_model = load_model(self.dataset_name).to(self.device)
            global_params = config.get("global_parameters", parameters)
            for param, new_param in zip(global_model.parameters(), global_params):
                param.data = torch.tensor(new_param, dtype=param.dtype)

            if self.attack_strategy == 'rl':    # RL adaptive attack
                loss_clean_before, acc_clean_before = self.evaluate_on_loader(self.model, self.test_benign_loader)
                loss_mal_before, acc_mal_before = self.evaluate_on_loader(self.model, self.test_malicious_loader)

                # RL poisoning: select action and poison
                state = self.poisoner.get_state(self.model.parameters())
                action = self.poisoner.model.select_action(state)
                print(f"[Client {self.cid}] Action taken: {action}")

                # apply rl adaptive poisoning attack
                poisoned_model = self.poisoner.poison(action, self.model)

                loss_clean_after, acc_clean_after = self.evaluate_on_loader(poisoned_model, self.test_benign_loader)
                loss_mal_after, acc_mal_after = self.evaluate_on_loader(poisoned_model, self.test_malicious_loader)

                # Compute next state and reward
                next_state = self.poisoner.get_state(poisoned_model.parameters())
                reward = self.poisoner.compute_reward(acc_clean_before, acc_clean_after, acc_mal_before, acc_mal_after)
                print(f"[Client {self.cid}] Clean acc: {acc_clean_before:.4f} → {acc_clean_after:.4f}, "
                      f"Mal acc: {acc_mal_before:.4f} → {acc_mal_after:.4f}, reward: {reward:.4f}")


                # Store and train RL agent
                #print(f"[RL] Storing: state={state}, action={action}, reward={reward}, next_state={next_state}")
                self.poisoner.model.store_transition(state, action, reward, next_state)
                print(f"[RL] Memory size after store: {len(self.poisoner.model.memory)}")

                self.poisoner.model.train_step()
                self.poisoner.decay()
                print(f"[Client {self.cid}] Epsilon: {self.poisoner.model.epsilon:.4f}")

                self.model = poisoned_model # assign updated model

            else:                           # normal adaptive attack
                print(f"[Client {self.cid}] Poisoning model.")
                self.model = self.poisoner.poison(global_model, self.model, current_round, total_rounds)

        new_params = self.get_parameters(config)
        update_norm = self.compute_update_norm(old_params, new_params)

        return new_params, len(self.train_loader.dataset), {'update_norm': float(update_norm)}

    def evaluate(self, parameters, config):
        """Perform evaluation."""
        self.set_parameters(parameters)
        loss, accuracy = self.evaluate_model()

        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'is_malicious': self.is_malicious,
        }
        return float(loss), int(len(self.test_loader.dataset)), metrics


def poisoned_client_fn(cid: str, dataset_type: str, dataset_name: str, is_malicious: bool, attack_strategy: str, seed=42) -> Poisoned_AdaptiveClient:
    """Factory function for creating client instances."""
    global _global_poisoner
    if _global_poisoner is None and is_malicious and attack_strategy == 'rl':
        _global_poisoner = RLAdaptiveModelPoisoning()

    return Poisoned_AdaptiveClient(
        cid=cid,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        is_malicious=is_malicious,
        attack_strategy=attack_strategy,
        seed=seed,
        poisoner=_global_poisoner if is_malicious and attack_strategy == 'rl' else None
    )

