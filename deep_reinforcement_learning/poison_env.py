from gym import Env, spaces
import numpy as np

class PoisonEnv(Env):
    def __init__(self, state_size=5, n_actions=3, max_rounds=20):
        super(PoisonEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)
        self.max_rounds = max_rounds
        self.round = 0
        self.current_state = np.zeros(state_size)

    def reset(self):
        self.round = 0
        self.current_state = np.random.rand(self.observation_space.shape[0])
        return self.current_state

    def step(self, action):
        # TODO: Replace these with real metrics from FL
        global_accuracy = np.random.rand()
        clean_accuracy = np.random.rand()
        malicious_accuracy = np.random.rand()
        last_loss = np.random.rand()

        reward = clean_accuracy - malicious_accuracy  # example reward
        self.round += 1
        done = self.round >= self.max_rounds

        self.current_state = np.array([
            global_accuracy,
            clean_accuracy,
            malicious_accuracy,
            last_loss,
            self.round / self.max_rounds
        ], dtype=np.float32)

        return self.current_state, reward, done, {}

    def render(self, mode="human"):
        print(f"[Round {self.round}] State: {self.current_state}")

    def close(self):
        pass
