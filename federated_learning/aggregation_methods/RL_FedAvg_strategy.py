from flwr.server.strategy import FedAvg
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes, FitRes, Scalar, Parameters

import numpy as np

from deep_reinforcement_learning.rl_defense.rl_defense import RLAggregationDefense

class RLDefenseStrategy(FedAvg):
    def __init__(self, total_rounds=50, total_clients=20, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self.rl_defense = RLAggregationDefense()
        self.evaluate_metrics_aggregation_fn = None
        self.total_clients = total_clients

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # Extract update norms and accuracies from client metrics
        update_norms = [res.metrics.get("update_norm", 0.0) for _, res in results]
        client_accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]


        # Apply RL logic to select which clients to keep
        selected_indices = self.rl_defense.select_clients(
            update_norms=update_norms,
            accuracies=client_accuracies,
            round=server_round,
        )

        # Filter results based on RL agent's decision
        filtered_results = [results[i] for i in selected_indices]

        # Log for debugging
        print(f"[Round {server_round}] Selected {len(filtered_results)} clients out of {len(results)}")

        # Proceed with standard FedAvg aggregation on selected clients
        return super().aggregate_fit(server_round, filtered_results, failures)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        clean_metrics = [res.metrics for _, res in results if not res.metrics.get("poisoned")]
        mal_metrics = [res.metrics for _, res in results if res.metrics.get("poisoned")]

        clean_acc = float(np.mean([m["accuracy"] for m in clean_metrics]) if clean_metrics else 0.0)
        clean_loss = float(np.mean([m["loss"] for m in clean_metrics]) if clean_metrics else 0.0)

        mal_acc = float(np.mean([m["accuracy"] for m in mal_metrics]) if mal_metrics else 0.0)
        mal_loss = float(np.mean([m["loss"] for m in mal_metrics]) if mal_metrics else 0.0)

        global_acc = float(np.mean([res.metrics["accuracy"] for _, res in results]))
        global_loss = float(np.mean([res.metrics["loss"] for _, res in results]))

        keep_rate = len(results) / self.total_clients if self.total_clients else 0.0
        update_norms = [res.metrics.get("update_norm", 0.0) for _, res in results]
        client_accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]


        return global_acc, {
            "clean_accuracy": clean_acc,
            "clean_loss": clean_loss,
            "malicious_accuracy": mal_acc,
            "malicious_loss": mal_loss,
            "accuracy": global_acc,
            "loss": global_loss,
            "update_norm": update_norms,
            "client_accuracies": client_accuracies,
            "keep_rate": keep_rate,
        }

