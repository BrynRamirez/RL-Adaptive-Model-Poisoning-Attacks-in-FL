import flwr as fl
from flwr.server.strategy import fedtrimmedavg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Scalar, EvaluateRes, FitRes, FitIns, Parameters

from typing import List, Tuple, Optional, Dict, Union

""" Trimmed Mean/Avg aggregation strategy """
class FedTrimmedAvg(fedtrimmedavg):
    def __init__(self,total_rounds=50, total_clients=20, beta=0.1, **kwargs):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=total_clients,
            min_available_clients=total_clients,
            beta=beta,
            **kwargs
        )
        self.total_rounds = total_rounds
        self.total_clients = total_clients
        self.evaluate_metrics_aggregation_fn = None

