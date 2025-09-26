import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, EvaluateRes, FitRes, FitIns, Parameters

import numpy as np
from typing import List, Tuple, Optional, Dict, Union

# FedAvg strategy that aggregates our clients
class AggregateCustomMetricStrategy(FedAvg):
    def __init__(self, total_rounds=50, total_clients=20, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self.evaluate_metrics_aggregation_fn = None
        self.total_clients = total_clients

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Inject round info and global model parameters into client config."""

        config = {
            "round_num": server_round,
            "total_rounds": self.total_rounds if hasattr(self, "total_rounds") else 50,
            "total_clients": self.total_clients if hasattr(self, "total_clients") else 20,
        }

        fit_ins = FitIns(parameters, config)

        sampled_clients = client_manager.sample(
            num_clients=self.min_fit_clients or 2,
            min_num_clients=self.min_available_clients or 2,
        )

        return [(client, fit_ins) for client in sampled_clients]

    # Local Test Datasets
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract norms and weights
        update_norms = []
        weights = []

        for _, res in results:
            norm = res.metrics.get("update_norm", 0.0)
            weight = res.num_examples
            update_norms.append(norm * weight)
            weights.append(weight)

        # Compute weighted average
        weighted_avg_norm = sum(update_norms) / sum(weights) if weights else 0.0

        print(f"[Round {server_round}] Weighted Avg Update Norm: {weighted_avg_norm:.4f}")

        # Use FedAvg to aggregate weights
        aggregated_parameters, _ = super().aggregate_fit(server_round, results, failures)

        return aggregated_parameters, {
            "avg_update_norm": weighted_avg_norm,
            "update_norms": update_norms,  # Optional: raw list for analysis
        }

    # Global Test Datasets
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics, separated by client type."""

        if not results:
            return None, {}
        # Separate clean and malicious metrics
        clean_accuracies, malicious_accuracies = [], []
        clean_losses, malicious_losses = [], []
        clean_weights, malicious_weights = [], []

        total_accuracy, total_loss = [], []
        total_update_norms = []
        total_client_acc = []
        total_weights = []

        for _, res in results:
            acc = res.metrics["accuracy"]
            loss = res.metrics["loss"]
            norms = res.metrics.get("update_norm", 0.0)
            weight = res.num_examples
            is_malicious = res.metrics.get("is_malicious", False)

            if is_malicious:
                malicious_accuracies.append(acc * weight)
                malicious_losses.append(loss * weight)
                malicious_weights.append(weight)
            else:
                clean_accuracies.append(acc * weight)
                clean_losses.append(loss * weight)
                clean_weights.append(weight)

            total_accuracy.append(acc * weight)
            total_loss.append(loss * weight)
            total_update_norms.append(norms)
            total_client_acc.append(acc * weight)
            total_weights.append(weight)

        def weighted_avg(values, weights):
            return sum(values) / sum(weights) if weights else 0.0

        clean_acc = weighted_avg(clean_accuracies, clean_weights)
        mal_acc = weighted_avg(malicious_accuracies, malicious_weights)
        clean_loss = weighted_avg(clean_losses, clean_weights)
        mal_loss = weighted_avg(malicious_losses, malicious_weights)

        global_acc = weighted_avg(total_accuracy, total_weights)
        global_loss = weighted_avg(total_loss, total_weights)

        update_norms = [res.metrics.get("update_norm", 0.0) for _, res in results]
        client_accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
        keep_rate = len(results) / self.total_clients if self.total_clients else 0.0


        print(f"[Round {server_round}] Train accuracy: {global_acc:.4f}")
        print(f"[Round {server_round}] Train loss: {global_loss:.4f}")
        print(f"[Round {server_round}] Train Clean clients avg accuracy: {clean_acc:.4f}")
        print(f"[Round {server_round}] Train Malicious clients avg accuracy: {mal_acc:.4f}")

        return global_acc, {
            "accuracy": global_acc,
            "loss": global_loss,
            "clean_accuracy": clean_acc,
            "clean_loss": clean_loss,
            "malicious_accuracy": mal_acc,
            "malicious_loss": mal_loss,
            #"update_norm": update_norms,
            "client_accuracies": client_accuracies,
            "keep_rate": keep_rate,
        }