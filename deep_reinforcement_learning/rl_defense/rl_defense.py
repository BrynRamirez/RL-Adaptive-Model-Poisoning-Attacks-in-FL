import numpy as np
from deep_reinforcement_learning.rl_defense.dqn_agent import FLDQNAgent

class RLAggregationDefense:
    def __init__(self, state_dim=5, action_dim=2):
        self.agent = FLDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.01,
            gamma=0.95,
            lr=3e-4,
        )
        self.last_global_accuracy = 0.0
        self.last_client_accuracies = []
        self.round_num = 0

    def select_clients(self, update_norms, accuracies, round):
        # Example: reward = accuracy - penalty on update norm
        rewards = []
        for acc, norm in zip(accuracies, update_norms):
            reward = acc - 0.1 * norm  # simple reward function
            rewards.append(reward)

        # Rank clients by reward
        ranked_clients = sorted(enumerate(rewards), key=lambda x: x[1], reverse=True)

        # Keep top X% clients (e.g., top 70%)
        keep_ratio = 0.7
        num_keep = max(1, int(len(ranked_clients) * keep_ratio))
        selected_indices = [idx for idx, _ in ranked_clients[:num_keep]]

        return selected_indices

    def create_state(self, metrics):
        acc = metrics.get('accuracy', 0.0)
        loss = metrics.get('loss', 0.0)
        weight = metrics.get('weight', 1.0)
        update_norm = metrics.get('update_norm', 0.0)
        confidence = max(min(acc - loss, 1.0), -1.0)
        state = np.array([acc, loss, weight, update_norm, confidence], dtype=np.float32)
        return state


    def decide(self, metrics):
        state = self.create_state(metrics)
        action = self.agent.select_action(state)
        return action, state

    def reward(self, global_metrics):
        global_accuracy = global_metrics.get('accuracy', 0.0)
        client_accuracies = global_metrics.get('client_accuracies', [])
        loss = global_metrics.get('loss', 0.0)
        update_norm = np.mean(global_metrics.get("client_update_norms", [])) if global_metrics.get("client_update_norms") else 0.0
        keep_rate = global_metrics.get("keep_rate", 1.0)

        update_norm = update_norm / (1.0 + update_norm)


        # penalize instability/unpredictable accuracy swings
        if self.round_num == 0:
            volatile_penalty = 0.0 # skip penalty on first round
        else:
            volatile_penalty = abs(global_accuracy - self.last_global_accuracy)
        # penalize inconsistent client results
        inconsistency_penalty = np.std(client_accuracies) if client_accuracies else 0.0

        reward = (
                global_accuracy
                  - 0.2 * volatile_penalty
                  - 0.3 * inconsistency_penalty #
                  - 0.2 * loss                  # coefficient represents the importance of loss in the reward
                  - 0.1 * update_norm
                  + 0.2 *  keep_rate
        )

        reward = max(min(reward, 1.0), -1.0) # reward clipping
        print(
            f"[RL REWARD] acc={global_accuracy:.4f}, vol={volatile_penalty:.4f}, loss={loss:.4f}, inc={inconsistency_penalty:.4f}, norm={update_norm:.4f}, keep_rate={keep_rate:.2f}, reward={reward:.4f}")

        self.last_global_accuracy = global_accuracy
        self.last_client_accuracies = client_accuracies
        return reward

    def train_on_round(self, client_state, client_actions, global_metrics):
        print(f"[RL DEFENSE] Training RL agent on round {self.round_num}", flush=True)
        if self.round_num > 1:
            self.agent.epsilon = max(0.01, self.agent.epsilon * 0.095)
        else:
            self.agent.epsilon = 0.0

        reward = self.reward(global_metrics)

        for state, action in zip(client_state, client_actions):
            next_state = state
            self.agent.store_transition(state, action, reward, next_state)
        self.agent.train_step()

        self.round_num += 1

