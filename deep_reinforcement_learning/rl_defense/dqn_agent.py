import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

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
class FLDQNAgent:
    def __init__(self, state_dim, action_dim, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.policy_net = DQNNet(state_dim, action_dim)
        self.target_net = DQNNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.batch_size = 32
        self.update_target_steps = 100
        self.train_step_count = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)

        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze()
        next_q_values = self.target_net(next_state).max(1)[0].detach()
        target_q_values = reward + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def local_train(self, num_steps=1):
        for _ in range(num_steps):
            self.train_step()

    def evaluate_model(self, env_fn):
        total_reward = 0
        episodes = 2
        for _ in range(episodes):
            state = env_fn.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.select_action(state)
                state, reward, done = env_fn.step(action)
                ep_reward += reward
            total_reward += ep_reward
        return total_reward / episodes

    def get_weights(self):
        return {k: v.cpu().clone() for k, v in self.policy_net.state_dict().items()}

    def set_weights(self, weights):
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)

    def get_gradients(self):
        grads = {}
        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().clone()
        return grads

    def apply_gradients(self, gradients):
        for name, param in self.policy_net.named_parameters():
            if name in gradients:
                param.grad = gradients[name].to(param.device)
        self.optimizer.step()
        self.optimizer.zero_grad()

