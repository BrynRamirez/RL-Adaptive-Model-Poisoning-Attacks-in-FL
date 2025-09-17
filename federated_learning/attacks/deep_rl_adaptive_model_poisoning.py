import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque



class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size=5, action_size=5, lr=1e-3, gamma=0.95, buffer_size=10000, batch_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = DQNNet(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)

        self.epsilon = 0.5  # start with pure exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            action = random.randint(0, self.action_size - 1)
            print(f"[RL] Exploring (eps={self.epsilon:.4f}): action={action}")
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                print(f"[RL] Q-values: {q_values}")
                action = torch.argmax(q_values, dim=1).item()
                print(f"[RL] Exploiting: q_values={q_values.cpu().numpy()}, action={action}")
                return action


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_step(self):
        print(f'[RL] Memory size: {len(self.memory)}')  # should increase
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_values = self.model(states).gather(1, actions).squeeze()
        next_q_values = self.model(next_states).max(1)[0].detach()
        targets = rewards + self.gamma * next_q_values

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # epsilon decay
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # log training progress
        print(f'[RL] Loss: {loss.item():.6f}')
        print(f'[RL] Epsilon after decay: {self.epsilon:.4f}')


class RLAdaptiveModelPoisoning:
    """Reinforcement Learning based Adaptive Model Poisoning Attack: attack type only noise-based poisoning"""
    def __init__(self, state_size=5, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNAgent(state_size=state_size, action_size=action_size)
        self.last_state = None
        self.last_action = None
        self.last_acc_clean = None
        self.last_acc_mal = None

    def get_state(self, model_params):
        # Flatten or summarize model params to feed into RL agent
        flat_state = []
        for param in model_params:
            flat_state.append(param.mean().item())
            flat_state.append(param.std().item())
        # Pad/truncate to match state_size
        flat_state = flat_state[:self.state_size] + [0.0] * max(0, self.state_size - len(flat_state))
        print(f"[RL] State: {flat_state}")
        return np.array(flat_state, dtype=np.float32)

    def select_action(self, state):
        return self.model.select_action(state, self.model.epsilon)

    def decay(self):
        self.model.decay_epsilon()

    def poison(self, action, model):
        # Define how action maps to poisoning strength or strategy
        poisoning_strengths = [0.0, 0.01, 0.02, 0.03, 0.04]
        strength = poisoning_strengths[action]
        print(f"[RL] Action: {action}, Strength: {strength}")

        # Apply noise-based poisoning with selected strength
        poisoned_params = []
        for param in model.parameters():
            noise = torch.randn_like(param) * strength
            poisoned_params.append(param + noise)

        # Replace parameters in model
        for p, new_p in zip(model.parameters(), poisoned_params):
            p.data = new_p.clone()

        return model

    def compute_reward(self, acc_clean_before, acc_clean_after, acc_mal_before, acc_mal_after):
        print(f"[RL] Clean Acc: {acc_clean_before:.4f} → {acc_clean_after:.4f}, ")
        gain = acc_mal_after - acc_mal_before
        loss = acc_clean_after - acc_clean_before

        # Configuration
        penalty_threshold = -0.0005  # penalize at 0.05% drop
        penalty = 0.0

        if loss < penalty_threshold:
            # exponential penalty
            penalty = 10 * np.exp(100 * abs(loss))

        raw_reward = gain - loss - penalty

        # soft clipping on reward
        reward = max(min(raw_reward, 0.01), -0.01)
        return reward

    def step(self, model_params, acc_clean_before, acc_clean_after, acc_mal_before, acc_mal_after):
        # Compute next state
        next_state = self.get_state(model_params)
        # Compute reward
        reward = self.compute_reward(acc_clean_before, acc_clean_after, acc_mal_before, acc_mal_after)
        # Store transition
        if self.last_state is not None and self.last_action is not None:
            self.model.store_transition(self.last_state, self.last_action, reward, next_state)
            self.model.train_step()
        # Update last state/action
        self.last_state = next_state


