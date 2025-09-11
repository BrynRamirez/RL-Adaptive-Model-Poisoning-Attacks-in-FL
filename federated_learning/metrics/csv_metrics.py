import pandas as pd
import os

def export_metrics_to_csv(histories, datasets, strategies, run_id="default_run", output_dir='federated_learning/Federated Metrics'):
    output_dir = os.path.join(output_dir, run_id, 'csv_metrics')
    os.makedirs(output_dir, exist_ok=True)

    def metrics_to_dataframe(metrics_dict):
        df = pd.DataFrame()
        for metric_name, values in metrics_dict.items():
            rounds = [r[0] for r in values]
            scores = [r[1] for r in values]
            df[metric_name] = pd.Series(data=scores, index=rounds)
        df.index.name = 'round'
        return df

    for dataset in datasets:
        for strategy in strategies:
            history = histories[dataset][strategy]
            prefix = f"{dataset}_{strategy}_{run_id}"

            dataset_dir = os.path.join(output_dir, f'{dataset}')
            os.makedirs(dataset_dir, exist_ok=True)

            # Train (distributed) metrics
            train_df = metrics_to_dataframe(history.metrics_distributed)
            train_df.to_csv(os.path.join(dataset_dir, f"{prefix}_train_metrics.csv"))

            # Test (centralized) metrics
            test_df = metrics_to_dataframe(history.metrics_centralized)
            test_df.to_csv(os.path.join(dataset_dir, f"{prefix}_test_metrics.csv"))

            # Losses - train
            if history.losses_distributed:
                loss_train_df = pd.DataFrame(history.losses_distributed, columns=['round', 'loss'])
                loss_train_df.to_csv(os.path.join(dataset_dir, f"{prefix}_train_losses.csv"), index=False)

            # Losses - test
            if history.losses_centralized:
                loss_test_df = pd.DataFrame(history.losses_centralized, columns=['round', 'loss'])
                loss_test_df.to_csv(os.path.join(dataset_dir, f"{prefix}_test_losses.csv"), index=False)

    print(f"[Saved] All CSVs to {output_dir}")
