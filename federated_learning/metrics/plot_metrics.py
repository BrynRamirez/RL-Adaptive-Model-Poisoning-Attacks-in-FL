import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_comparisons(all_histories, datasets, strategies, run_id="run-xxxx"):
    n_rows = len(strategies)
    n_cols = len(datasets)

    fig_acc, axes_acc = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_loss, axes_loss = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_client_acc, axes_client_acc = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_malicious_client_acc, axes_malicious_client_acc = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_update_norm, axes_update_norm = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_keep_rate, axes_keep_rate = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)

    for row_idx, strategy in enumerate(strategies):
        for col_idx, dataset in enumerate(datasets):
            history = all_histories.get(dataset, {}).get(strategy)
            if history is None:
                continue

            train_acc = history.metrics_distributed.get("accuracy", [])
            train_loss = history.metrics_distributed.get("loss", [])

            clean_acc = history.metrics_distributed.get("clean_accuracy", [])
            clean_loss = history.metrics_distributed.get("clean_loss", [])

            malicious_acc = history.metrics_distributed.get("malicious_accuracy", [])
            malicious_loss = history.metrics_distributed.get("malicious_loss", [])

            client_acc = history.metrics_distributed.get("client_accuracies", [])
            client_update_norms = history.metrics_distributed_fit.get("update_norms", [])
            client_avg_update_norm = history.metrics_distributed_fit.get("avg_update_norm", [])
            keep_rate = history.metrics_distributed.get("keep_rate", [])


            # accuracies
            ax_acc = axes_acc[row_idx][col_idx]
            if train_acc:
                ax_acc.plot([r[0] for r in train_acc], [r[1] for r in train_acc], label ='Global Acc', marker="o", color="black")
            if clean_acc:
                ax_acc.plot([r[0] for r in clean_acc], [r[1] for r in clean_acc], label="Benign Acc",
                            linestyle="--", color='blue')
            if malicious_acc:
                ax_acc.plot([r[0] for r in malicious_acc], [r[1] for r in malicious_acc], label="Malicious Acc",
                            linestyle=":", color='red')

            ax_acc.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_acc.set_xlabel("Rounds")
            ax_acc.set_ylabel("Train Accuracy")
            ax_acc.legend()

            # losses
            ax_loss = axes_loss[row_idx][col_idx]
            if train_loss:
                ax_loss.plot([r[0] for r in train_loss], [r[1] for r in train_loss], label="Train Loss", marker='o', color="black")
            if clean_loss:
                ax_loss.plot([r[0] for r in clean_loss], [r[1] for r in clean_loss], label="Benign Train Loss", color="blue")
            if malicious_loss:
                ax_loss.plot([r[0] for r in malicious_loss], [r[1] for r in malicious_loss], label="Malicious Train Loss", color="red")
            ax_loss.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_loss.set_xlabel("Rounds")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()

            # per-client accuracy
            num_clients = len(client_acc[0][1])
            client_acc_series = [[] for _ in range(num_clients)]
            for round_idx, accuracies in client_acc:
                for cid, acc in enumerate(accuracies):
                    client_acc_series[cid].append((round_idx, acc))

            ax_client_acc = axes_client_acc[row_idx][col_idx]
            for cid, acc_data in enumerate(client_acc_series):
                rounds = [r[0] for r in acc_data]
                accs = [r[1] for r in acc_data]
                ax_client_acc.plot(rounds, accs, label=f"Client {cid}")
            ax_client_acc.set_title(f"{dataset.upper()} - {strategy.upper()} - Per Train Client Accuracy")
            ax_client_acc.set_xlabel("Rounds")
            ax_client_acc.set_ylabel("Accuracy")
            ax_client_acc.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')

            # per-client malicious accuracy
            num_clients = len(client_acc[0][1])
            client_malicious_acc_series = [[] for _ in range(num_clients)]
            for round_idx, accuracies in client_acc:
                for cid, acc in enumerate(accuracies):
                    client_malicious_acc_series[cid].append((round_idx, acc))

            ax_malicious_client_acc = axes_malicious_client_acc[row_idx][col_idx]
            # Plot benign clients first in background
            for cid, acc_data in enumerate(client_malicious_acc_series):
                rounds = [r[0] for r in acc_data]
                accs = [r[1] for r in acc_data]
                if cid % 5 != 0:  # non-malicious clients
                    ax_malicious_client_acc.plot(rounds, accs, alpha=0.35, linewidth=1)
                else:
                    ax_malicious_client_acc.plot(rounds, accs, linewidth=1,  label=f"Malicious Client {cid}")


            ax_malicious_client_acc.set_title(f"{dataset.upper()} - {strategy.upper()} - Per Train Malicious Client Accuracy")
            ax_malicious_client_acc.set_xlabel("Rounds")
            ax_malicious_client_acc.set_ylabel("Accuracy")
            ax_malicious_client_acc.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')

            # client update norms
            ax_update_norm = axes_update_norm[row_idx][col_idx]
            ax_update_norm.plot([r[0] for r in client_avg_update_norm], [r[1] for r in client_avg_update_norm], label="Client Update Norm", color="black")
            ax_update_norm.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_update_norm.set_xlabel("Rounds")
            ax_update_norm.set_ylabel("Client Average Update Norm")
            ax_update_norm.legend()

            # keep rate
            ax_keep_rate = axes_keep_rate[row_idx][col_idx]
            ax_keep_rate.plot([r[0] for r in keep_rate], [r[1] for r in keep_rate], label="Keep Rate", color="black")
            ax_keep_rate.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_keep_rate.set_xlabel("Rounds")
            ax_keep_rate.set_ylabel("Keep Rate")
            ax_keep_rate.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join("federated_learning", "Federated Metrics", run_id, 'graph_metrics')
    os.makedirs(out_dir, exist_ok=True)
    fig_acc.savefig(os.path.join(out_dir, "grid_train_accuracy.png"))
    fig_loss.savefig(os.path.join(out_dir, "grid_train_loss.png"))
    fig_client_acc.savefig(os.path.join(out_dir, "grid_train_client_accuracy.png"))
    fig_malicious_client_acc.savefig(os.path.join(out_dir, "grid_train_malicious_client_accuracy.png"))
    fig_update_norm.savefig(os.path.join(out_dir, "grid_train_avg_update_norm.png"))
    fig_keep_rate.savefig(os.path.join(out_dir, "grid_train_keep_rate.png"))
    print(f"[Saved] Train Accuracy and Loss grids to: {out_dir}")

# side-by-side comparison for testing metrics
def plot_test_comparisons(all_histories, datasets, strategies, run_id="run-xxxx"):
    n_rows = len(strategies)
    n_cols = len(datasets)

    fig_acc, axes_acc = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)
    fig_loss, axes_loss = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False, constrained_layout=True)

    for row_idx, strategy in enumerate(strategies):
        for col_idx, dataset in enumerate(datasets):
            history = all_histories.get(dataset, {}).get(strategy)
            if history is None:
                continue

            central_global_acc = history.metrics_centralized.get("central_global_accuracy", [])
            central_clean_acc = history.metrics_centralized.get("central_clean_accuracy", [])
            central_malicious_acc = history.metrics_centralized.get("central_malicious_accuracy", [])
            central_loss = history.losses_centralized if hasattr(history, 'losses_centralized') else []
            central_clean_loss = history.metrics_centralized.get("central_clean_loss", [])
            central_malicious_loss = history.metrics_centralized.get("central_malicious_loss", [])

            ax_acc = axes_acc[row_idx][col_idx]
            if central_global_acc:
                ax_acc.plot([r[0] for r in central_global_acc], [r[1] for r in central_global_acc], label="Global",
                            marker='o', color="black")
            if central_clean_acc:
                ax_acc.plot([r[0] for r in central_clean_acc], [r[1] for r in central_clean_acc], label="Benign",
                            linestyle="--", linewidth=2, color="blue")
            if central_malicious_acc:
                ax_acc.plot([r[0] for r in central_malicious_acc], [r[1] for r in central_malicious_acc],
                            label="Malicious", linestyle=":", linewidth=2, color="red")

            ax_acc.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_acc.set_xlabel("Rounds")
            ax_acc.set_ylabel("Test Accuracy")
            ax_acc.legend()

            ax_loss = axes_loss[row_idx][col_idx]
            if central_loss:
                ax_loss.plot([r[0] for r in central_loss], [r[1] for r in central_loss],
                             label="Test Loss",  color="black")
            if central_clean_loss:
                ax_loss.plot([r[0] for r in central_clean_loss], [r[1] for r in central_clean_loss],
                             label="Benign Test Loss", linestyle="--", color="blue")
            if central_malicious_loss:
                ax_loss.plot([r[0] for r in central_malicious_loss], [r[1] for r in central_malicious_loss],
                             label="Malicious Test Loss", linestyle=":", color="red")

            ax_loss.set_title(f"{dataset.upper()} - {strategy.upper()}")
            ax_loss.set_xlabel("Rounds")
            ax_loss.set_ylabel("Test Loss")
            ax_loss.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.join("federated_learning", "Federated Metrics", run_id, 'graph_metrics')
    os.makedirs(out_dir, exist_ok=True)
    fig_acc.savefig(os.path.join(out_dir, "grid_test_accuracy.png"))
    fig_loss.savefig(os.path.join(out_dir, "grid_test_loss.png"))
    print(f"[Saved] Test Accuracy and Loss grids to: {out_dir}")

def plot_krum_scores_over_rounds(krum_scores_per_round, selected_indices_per_round, run_id="run-xxxx", dataset="dataset", attack_strategy="attack", num_clients=20):
    """
    krum_scores_per_round: list of lists, each inner list is Krum scores for all clients in a round
    selected_indices_per_round: list of lists, each inner list is indices of selected clients for that round
    """
    num_rounds = len(krum_scores_per_round)
    num_clients = len(krum_scores_per_round[0]) if num_rounds > 0 else 0
    selected_indices_per_round = np.array(selected_indices_per_round) #shape: (num_rounds, selected client indices for that round)
    krum_scores = np.array(krum_scores_per_round)  # shape: (num_rounds, num_clients)

    out_dir = os.path.join("federated_learning", "Federated Metrics", run_id, 'graph_metrics', dataset)
    os.makedirs(out_dir, exist_ok=True)

    # Heatmap
    plt.figure(figsize=(14, 14))
    plt.imshow(krum_scores.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Krum Score')
    plt.xlabel('Round')
    plt.ylabel('Client Index')
    plt.title(f'{dataset.upper()} {attack_strategy.upper()} Krum Scores Heatmap (Clients x Rounds)')
    plt.tight_layout()
    plt.xticks(ticks=np.arange(num_rounds), labels=np.arange(0, num_rounds))
    plt.yticks(ticks=np.arange(num_clients), labels=np.arange(0, num_clients))
    plt.savefig(os.path.join(out_dir, f"{dataset}_{attack_strategy}_krum_scores_heatmap.png"))
    plt.close()

    # dot plot for selected clients round and krum scores
    plt.figure(figsize=(14, 10))
    for r in range(num_rounds):
        for idx in selected_indices_per_round[r]:
            plt.plot(r, krum_scores[r, idx], 'go')
    plt.xlabel('Round')
    plt.ylabel('Krum Score')
    plt.title(f'{dataset.upper()} {attack_strategy.upper()} Selected Clients\' Krum Scores Over Rounds')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset}_{attack_strategy}_krum_scores_selected_clients.png"))
    plt.close()

    # dot plot for selected clients and rounds
    plt.figure(figsize=(14, 10))
    rounds = []
    clients = []
    for r, selected in enumerate(selected_indices_per_round):
        for client in selected:
            rounds.append(r)
            clients.append(client)
    plt.scatter(rounds, clients, label='Selected Clients', marker='o')
    plt.xlabel('Round')
    plt.ylabel('Client Index')
    plt.title(f'{dataset.upper()} {attack_strategy.upper()} Selected Clients Over Rounds')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset}_{attack_strategy}_selected_clients_over_rounds.png"))
    plt.close()

