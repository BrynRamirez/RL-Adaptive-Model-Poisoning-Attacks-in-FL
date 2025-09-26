from flwr.server import ClientManager
from flwr.server.strategy.krum import Krum
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, EvaluateRes, FitRes, FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays, \
    EvaluateIns

import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Any

""" 
Multi-Krum aggregation strategy that selects m most trustworthy clients
    and averages their updates for Byzantine-robust federated learning
"""
class MultiKrum(Krum):
    def __init__(self, total_rounds=50, total_clients=20, **kwargs):
        super().__init__(**kwargs)
        self.fraction_fit=1.0
        self.fraction_evaluate=1.0
        self.min_fit_clients = total_clients
        self.min_evaluate_clients = total_clients
        self.min_available_clients = total_clients
        self.total_rounds = total_rounds
        self.total_clients = total_clients
        self.num_clients = total_clients
        self.num_malicious_clients = 4  # 20 % 5 = 4
        self.num_clients_to_keep = self.num_clients - self.num_malicious_clients
        self.krum_scores_per_round = []
        self.selected_indices_per_round = []


    def configure_fit(self, server_round: int, parameters, client_manager):
        """Inject round info and global model parameters into client config."""

        config = {
            "round_num": server_round,
            "total_rounds": self.total_rounds if hasattr(self, "total_rounds") else 50,
            "total_clients": self.total_clients if hasattr(self, "total_clients") else 20,
        }

        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        sampled_clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients= min_num_clients,
        )

        return [(client, fit_ins) for client in sampled_clients]

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Krum(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        if not results:
            return None, {}
        # Does not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # extract parameters and weights from results
        parameters_list = []
        weights_list = []
        client_ids = []

        for client_proxy, fit_res in results:
            parameters_list.append(parameters_to_ndarrays(fit_res.parameters))
            weights_list.append(fit_res.num_examples)
            client_ids.append(client_proxy.cid)

        selected_params, krum_scores, selected_indices = self._multikrum_aggregate(parameters_list, client_ids)
        if selected_params is None:
            print(f"Multi-Krum aggregation failed.")
            return None, {}

        # Store for plotting
        self.krum_scores_per_round.append(krum_scores)
        self.selected_indices_per_round.append(selected_indices)

       # convert aggregated parameters back to Parameters
        aggregated_parameters = ndarrays_to_parameters(selected_params)

        # aggregate metrics
        metrics_aggregated = self._aggregate_metrics(results)
        metrics_aggregated["aggregation_method"] = "multi_krum"
        metrics_aggregated["selected_clients"] = self.num_clients
        metrics_aggregated["krum_scores"] = krum_scores
        metrics_aggregated["krum_selected_indices"] = selected_indices

        return aggregated_parameters, metrics_aggregated

    """Compute Multi-Krum aggregation of all weights"""
    def _multikrum_aggregate(self,
        parameters_list: List[List[np.ndarray]],
        client_ids: List[str]
    ) -> Optional[List[np.ndarray]]:

        n = len(parameters_list)

        # check if we have enough clients to perform Multi-Krum
        if n <= 2 * self.num_malicious_clients:
            print(f"Number of clients {n} is too small for Multi-Krum with {self.num_malicious_clients} malicious clients.")
            return None

        # Ensure num_clients_to_keep doesn't exceed available non-Byzantine clients
        max_num_clients_to_keep = n - self.num_malicious_clients
        actual_num_clients = min(self.num_clients_to_keep, max_num_clients_to_keep)

        print(f"Multi-Krum: n={n}, num_malicious_clients={self.num_malicious_clients}, "
              f"num_clients_to_keep={actual_num_clients}")

        # flatten parameters for distance computation
        flattened_parameters = []
        original_shapes = []

        # store original shapes for reshaping later
        if parameters_list:
            original_shapes =[param.shape for param in parameters_list[0]]

        for params in parameters_list:
            flat_param = np.concatenate([p.flatten() for p in params])
            flattened_parameters.append(flat_param)
        flattened_parameters = np.array(flattened_parameters)

        # compute pairwise squared Euclidean distances
        print(f"Calculating pairwise distances...")
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(flattened_parameters[i] - flattened_parameters[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        # compute krum score
        krum_score = n - self.num_malicious_clients - 2 # number of closest neighbors to consider
        if krum_score <= 0:
            print(f"Krum score {krum_score} is not positive. Adjust num_malicious_clients or ensure enough clients.")
            return None

        print(f"Computing Krum scores with krum_score = {krum_score}...")
        scores = np.zeros(n)
        for i in range(n):
            # Get distances to all other clients (excluding self)
            neighbor_distances = np.delete(distances[i], i)
            # Find the krum_score smallest distances
            closest_distances = np.partition(neighbor_distances, krum_score-1)[:krum_score]
            scores[i] = np.sum(closest_distances)

        # select clients with lowest scores
        selected_indices = np.argsort(scores)[:actual_num_clients]
        selected_client_ids = [int(client_ids[i]) for i in selected_indices]
        selected_scores = [scores[i] for i in selected_indices]

        for idx, score in zip(selected_indices, selected_scores):
            print(f"Selected index: {idx}, Selected score: {score}")

        # get selected client parameters
        selected_parameters = [parameters_list[i] for i in selected_indices]

        # compute weighted average of selected clients
        weights = np.ones(actual_num_clients) / actual_num_clients
        print(f"Averageing {actual_num_clients} selected client updates with weights: {weights}")

        # perform weighted aggregation
        aggregated = []
        for layer_idx in range(len(selected_parameters[0])):  # Iterate through layers
            layer_sum = np.zeros_like(selected_parameters[0][layer_idx])
            for client_idx, weight in enumerate(weights):
                layer_sum += weight * selected_parameters[client_idx][layer_idx]
            aggregated.append(layer_sum)

        return aggregated, scores.tolist(), selected_indices.tolist()

    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict[str, Scalar]:
        """Aggregate metrics from training results"""
        metrics = {}

        # collect all metrics
        all_metrics = {}
        total_examples = 0

        for _, fit_res in results:
            total_examples += fit_res.num_examples
            if fit_res.metrics:
                for key, value in fit_res.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                        all_metrics[key].append((value, fit_res.num_examples))
        # weighted average
        for key, value_weight_pairs in all_metrics.items():
            if isinstance(value_weight_pairs[0][0], (int, float)):
                weighted_sum = sum(value * weight for value, weight in value_weight_pairs)
                metrics[f"weighted_avg_{key}"] = weighted_sum / total_examples

                simple_avg = sum(value for value, _ in value_weight_pairs) / len(value_weight_pairs)
                metrics[f"simple_avg_{key}"] = simple_avg

        metrics["total_examples"] = total_examples
        metrics["num_clients"] = len(results)
        return metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate==0.0:
            return []
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # return config paris
        config = {
            "round_num": server_round,
            "total_rounds": self.total_rounds if hasattr(self, "total_rounds") else 50,
            "total_clients": self.total_clients if hasattr(self, "total_clients") else 20
        }
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Any]]:
        """Aggregate evaluation metrics"""
        if not results:
            print("No evaluation results to aggregate.")
            return None, {}

        benign_accuracy, malicious_accuracy = [], []
        benign_loss, malicious_loss = [], []
        benign_weights, malicious_weights = [], []

        total_accuracy, total_loss = [], []
        total_update_norms = []
        total_client_acc = []
        total_weights = []

        for _, evaluate_res in results:
            acc = evaluate_res.metrics["accuracy"]
            loss = evaluate_res.loss
            weight = evaluate_res.num_examples
            is_malicious = evaluate_res.metrics.get("is_malicious", False)

            if is_malicious:
                malicious_accuracy.append(acc * weight)
                malicious_loss.append(loss * weight)
                malicious_weights.append(weight)
            else:
                benign_accuracy.append(acc * weight)
                benign_loss.append(loss * weight)
                benign_weights.append(weight)

            total_accuracy.append(acc * weight)
            total_loss.append(loss * weight)
            total_weights.append(weight)

        def weighted_avg(values, weights):
            return sum(values) / sum(weights) if weights else 0.0

        clean_acc = weighted_avg(benign_accuracy, benign_weights)
        mal_acc = weighted_avg(malicious_accuracy, malicious_weights)
        clean_loss = weighted_avg(benign_loss, benign_weights)
        mal_loss = weighted_avg(malicious_loss, malicious_weights)

        global_acc = weighted_avg(total_accuracy, total_weights)
        global_loss = weighted_avg(total_loss, total_weights)

        update_norms = [res.metrics.get("update_norm", 0.0) for _, res in results]
        client_accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
        keep_rate = len(results) / self.total_clients if self.total_clients else 0.0


        print(f"[Round {server_round}] Train accuracy: {global_acc:.4f}")
        print(f"[Round {server_round}] Train loss: {global_loss:.4f}")
        print(f"[Round {server_round}] Train Benign clients avg accuracy: {clean_acc:.4f}")
        print(f"[Round {server_round}] Train Malicious clients avg accuracy: {mal_acc:.4f}")

        return global_acc, {
            "accuracy": global_acc,
            "loss": global_loss,
            "clean_accuracy": clean_acc,
            "clean_loss": clean_loss,
            "malicious_accuracy": mal_acc,
            "malicious_loss": mal_loss,
            "update_norm": update_norms,
            "client_accuracies": client_accuracies,
            "keep_rate": keep_rate,
        }

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the number of clients used during fit."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the number of clients used during evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients
