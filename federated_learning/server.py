import flwr as fl

import logging, os, time
from datetime import datetime
import gc
import torch

from federated_learning.client import AdaptiveClient
from federated_learning.poisoned_client import poisoned_client_fn
from federated_learning.metrics.plot_metrics import plot_test_comparisons, plot_training_comparisons
from federated_learning.metrics.csv_metrics import export_metrics_to_csv

from federated_learning.aggregation_methods.RL_FedAvg_strategy import RLDefenseStrategy
from federated_learning.aggregation_methods.FedAvg_strategy import AggregateCustomMetricStrategy
from federated_learning.aggregation_methods.TrimmedMean_strategy import FedTrimmedAvg

# evaluate test clients
def centralized_test_evaluate(server_round, parameters, config):
    dataset_name = config.get('dataset_name', 'mnist')
    dataset_type = config.get("dataset_type", 'iid')

    central_client = AdaptiveClient(cid='0', dataset_name=dataset_name, dataset_type=dataset_type)
    central_client.set_parameters(parameters)
    central_client.model.to(central_client.device)

    metrics = central_client.evaluate_model()

    # Log the metrics
    print(f"[Round {server_round}] Central Evaluation Metrics:")
    for k, v in metrics.items():
        if "accuracy" in k:
            print(f" - {k}: {v:.4f}")

    # Return global loss + accuracy for the strategy,
    # plus all other metrics as additional info
    return metrics["central_global_loss"], metrics

def start_federated_simulation(num_rounds=50, num_clients=20, dataset_type='', dataset_name='', attack_strategy='', aggregation_strategy=''):
    """Start a Flower federated learning simulation."""
    # Define strategy (FedAvg as default)
    if aggregation_strategy == 'FedAvg':
        print("[Aggregation Strategy]: FedAvg")
        strategy = AggregateCustomMetricStrategy(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            total_rounds=num_rounds,
            total_clients=num_clients,
            evaluate_fn=lambda r, p, c: centralized_test_evaluate(r, p, {"dataset_type": dataset_type,
                                                                           "dataset_name": dataset_name})
        )
    elif aggregation_strategy == 'TrimmedMean':
        print("[Aggregation Strategy]: TrimmedMean")
        strategy = FedTrimmedAvg(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            total_rounds=num_rounds,
            total_clients=num_clients,
            beta=0.2,  # trimming parameter
            evaluate_fn=lambda r, p, c: centralized_test_evaluate(r, p,{
                "dataset_type": dataset_type,
                "dataset_name": dataset_name
            })
        )
    elif aggregation_strategy == 'RL_FedAvg':
        print("[Aggregation Strategy]: RL_FedAvg")
        strategy = RLDefenseStrategy(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            total_rounds=num_rounds,
            total_clients=num_clients,
            evaluate_fn=lambda r, p, c: centralized_test_evaluate(r, p,{
                "dataset_type": dataset_type,
                "dataset_name": dataset_name
            })
        )
    else:           # default to FedAvg
        print("[Aggregation Strategy]: FedAvg")
        strategy = AggregateCustomMetricStrategy(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            total_rounds=num_rounds,
            total_clients=num_clients,
            evaluate_fn=lambda r, p, c: centralized_test_evaluate(r, p, {"dataset_type": dataset_type,
                                                                           "dataset_name": dataset_name})
        )

    # Configure logging
    log_dir = "federated_learning/simulation_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        f"fl_simulation_{dataset_type}_{dataset_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        filename=log_filename,  # Save logs to a file
        level=logging.INFO,  # Log level (INFO, DEBUG, ERROR, etc.)
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger()

    # log the history
    logger.info(f"Starting Federated Learning Simulation: dataset type: {dataset_type}, dataset name: {dataset_name},"
                 f" strategy: {strategy}, rounds: {num_rounds}, clients: {num_clients}")
    SEED = 42
    def client_fn(cid: str):
        """Use poisoned_client_fn for some clients."""
        # Convert client id to int (in case it's a string)
        client_id = int(cid)
        print(f"[CID]: {client_id}")

        if attack_strategy.lower() == 'none':
            is_malicious = False
        else:
            #20% of clients are
            is_malicious = client_id % 5 == 0

        return poisoned_client_fn(
            cid=cid,
            dataset_type=dataset_type,
            dataset_name=dataset_name,
            is_malicious=is_malicious,
            attack_strategy=attack_strategy,
            seed=SEED)

    # Run the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,  # Total number of clients
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return history

# Start the simulations
if __name__ == "__main__":
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

   # # MNIST Simulations
   #  mnist_simulation_start_time = time.time()
   #  print("Starting Baseline MNIST simulation using FedAvg.")
   #  mnist_baseline_history = start_federated_simulation(
   #      num_rounds=50,
   #      num_clients=20,
   #      dataset_type='iid',
   #      dataset_name='mnist',
   #      attack_strategy='none',
   #      aggregation_strategy='FedAvg',
   #  )
   #  # clear memory in between simulations
   #  gc.collect()
   #  torch.cuda.empty_cache()
   #
   #  time.sleep(1)
   #
   #  print("Starting Adaptive Scaling MNIST simulation using FedAvg.")
   #  mnist_scale_history = start_federated_simulation(
   #      num_rounds=50,
   #      num_clients=20,
   #      dataset_type='iid',
   #      dataset_name='mnist',
   #      attack_strategy='scale',
   #      aggregation_strategy='FedAvg',
   #  )
   #  # clear memory in between simulations
   #  gc.collect()
   #  torch.cuda.empty_cache()
   #
   #  time.sleep(1)

    print("Starting Adaptive Noise MNIST simulation.")
    mnist_noise_history = start_federated_simulation(
        num_rounds=50,
        num_clients=20,
        dataset_type='iid',
        dataset_name='mnist',
        attack_strategy='noise',
        aggregation_strategy='FedAvg'
    )
    # clear memory in between simulations
    gc.collect()
    torch.cuda.empty_cache()

    # print("Starting RL Adaptive MNIST simulation using FedAvg.")
    # mnist_rl_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='iid',
    #     dataset_name='mnist',
    #     attack_strategy='rl',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()

    # time.sleep(1)
    # mnist_simulation_end_time = time.time()
    # print('MNIST Simulation Time:', mnist_simulation_end_time - mnist_simulation_start_time)
    #
    # # CIFAR10 Simulations
    # cifar10_simulation_start_time = time.time()
    # print("Starting Baseline CIFAR10 simulation using FedAvg.")
    # cifar10_baseline_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='iid',
    #     dataset_name='cifar10',
    #     attack_strategy='none',
    #     aggregation_strategy = 'FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Scaling CIFAR10 simulation using FedAvg.")
    # cifar10_scale_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='iid',
    #     dataset_name='cifar10',
    #     attack_strategy='scale',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Noise CIFAR10 simulation using FedAvg.")
    # cifar10_noise_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='iid',
    #     dataset_name='cifar10',
    #     attack_strategy='noise',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)

    # print("Starting RL Adaptive  CIFAR10 simulation using FedAvg.")
    # cifar10_rl_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='iid',
    #     dataset_name='cifar10',
    #     attack_strategy='rl',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    # cifar10_simulation_end_time = time.time()
    # print('CIFAR10 Simulation Time:', cifar10_simulation_end_time - cifar10_simulation_start_time)
    #
    # # EMNIST Simulations
    # emnist_simulation_start_time = time.time()
    # print("Starting Baseline EMNIST simulation using FedAvg.")
    # emnist_baseline_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='emnist',
    #     attack_strategy='none',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Scaling EMNIST simulation using FedAvg.")
    # emnist_scale_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='emnist',
    #     attack_strategy='scale',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Noise EMNIST simulation using FedAvg.")
    # emnist_noise_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='emnist',
    #     attack_strategy='noise',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # print("Starting RL Adaptive EMNIST simulation using FedAvg.")
    # emnist_rl_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='emnist',
    #     attack_strategy='rl',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    # emnist_simulation_end_time = time.time()
    # print('EMNIST Simulation Time:', emnist_simulation_end_time - emnist_simulation_start_time)
    #
    # # CIFAR100 Simulations
    # cifar100_simulation_start_time = time.time()
    # print("Starting Baseline CIFAR100 simulation using FedAvg.")
    # cifar100_baseline_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='cifar100',
    #     attack_strategy='none',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Scaling CIFAR100 simulation using FedAvg.")
    # cifar100_scale_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='cifar100',
    #     attack_strategy='scale',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # time.sleep(1)
    #
    # print("Starting Adaptive Noise CIFAR100 simulation using FedAvg.")
    # cifar100_noise_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='cifar100',
    #     attack_strategy='noise',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # # print("Starting RL Adaptive CIFAR100 simulation using FedAvg.")
    # cifar100_rl_history = start_federated_simulation(
    #     num_rounds=50,
    #     num_clients=20,
    #     dataset_type='non_iid',
    #     dataset_name='cifar100',
    #     attack_strategy='rl',
    #     aggregation_strategy='FedAvg',
    # )
    # # clear memory in between simulations
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    #
    # time.sleep(1)
    # cifar100_simulation_end_time = time.time()
    # print('CIFAR100 Simulation Time:', cifar100_simulation_end_time - cifar100_simulation_start_time)
    #
    # print(f'MNIST Simulation Time: {mnist_simulation_end_time - mnist_simulation_start_time}\n'
    #       f'CIFAR10 Simulation Time: {cifar10_simulation_end_time - cifar10_simulation_start_time}\n'
    #       f'EMNIST Simulation Time: {emnist_simulation_end_time - emnist_simulation_start_time}\n'
    #       f'CIFAR100 Simulation Time: {cifar100_simulation_end_time - cifar100_simulation_start_time}\n'
    #       f'Total Time: {cifar100_simulation_end_time - mnist_simulation_start_time}\n')

    histories = {
        'mnist': {
            #'none': mnist_baseline_history,
            #'scale': mnist_scale_history,
            'noise': mnist_noise_history,
            #'rl': mnist_rl_history,
        },
        # 'cifar10': {
            # 'none': cifar10_baseline_history,
            # 'scale': cifar10_scale_history,
            # 'noise': cifar10_noise_history,
            # 'rl': cifar10_rl_history,
        # },
        # 'emnist': {
        #     'none': emnist_baseline_history,
        #     'scale': emnist_scale_history,
        #     'noise': emnist_noise_history,
        #     'rl': emnist_rl_history,
        # },
        # 'cifar100': {
        #     'none': cifar100_baseline_history,
        #     'scale': cifar100_scale_history,
        #     'noise': cifar100_noise_history,
        #     'rl': cifar100_rl_history,
        # }
    }
    # dynamically list datasets and strategies from histories{...}
    datasets = list(histories.keys())
    strategies = list(next(iter(histories.values())).keys())

    # plot training metrics
    plot_training_comparisons(histories, datasets, strategies,RUN_ID)

    #plot testing metrics
    plot_test_comparisons(histories, datasets, strategies,RUN_ID)

    # export testing and training metrics to csv
    export_metrics_to_csv(histories, datasets, strategies, RUN_ID)

