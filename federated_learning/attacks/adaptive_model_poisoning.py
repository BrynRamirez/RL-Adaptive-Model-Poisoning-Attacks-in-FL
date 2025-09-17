import torch
import copy
import numpy as np

class AdaptiveModelPoisoning():
    def __init__(self, strategy='scale',max_scale=0.05, noise_lower=0.001, noise_upper=0.1):
        self.strategy = strategy
        self.max_scale = max_scale
        self.noise_lower = noise_lower
        self.noise_upper = noise_upper

    def poison(self, global_model, client_model, round_num, total_rounds):
        poisoned_model = copy.deepcopy(client_model)

        if self.strategy == 'scale':
            scale_factor = self._get_scale_factor(round_num, total_rounds)
            for param_global, param_client in zip(global_model.parameters(), poisoned_model.parameters()):
                update = param_client.data - param_global.data
                param_client.data = param_global.data + scale_factor * update

        elif self.strategy == 'noise':
            noise_std = self._adaptive_noise(self.noise_upper, self.noise_lower, round_num, total_rounds)
            for param in poisoned_model.parameters():
                noise = torch.normal(mean=0.0, std=noise_std, size=param.data.size(), device=param.data.device)
                print(
                    f"Noise mean: {noise.mean().item():.4f}, std: {noise.std().item():.4f}, shape: {tuple(noise.shape)}")

                param.data += noise

        return poisoned_model

    def _get_scale_factor(self, round_num, total_rounds):
        """
            Adapt the scale factor as rounds progress
            Example: max_scale = 0.5
                (Round 11: when poisoning begins): 1 + (.5 - 1.0) * (11 / 50) = .89
                (Round 33: mid-poisoning) 1 + (0.5 - 1.0) * (33 / 50) = 0.669
                (Round 50: final round): 1 + (0.5 - 1.0) * (50 / 50) = 0.5
        """
        return 1.0 + (self.max_scale - 1.0) * (round_num / total_rounds)

    def _adaptive_noise(self, upper_bound, lower_bound, round_num, total_rounds):
        """
            Generate adaptive noise
            Examples:   (Round 11: when poisoning begins): 0.1 + 2.0 * (11 / 50) = 0.54
                        (Round 33: mid-poisoning) 0.1 + 2.0 * (33 / 50) = 1.42
                        (Round 50: final round): 0.1 + 2.0 * (50 / 50) = 2.1
        """
        noise_strength = lower_bound + (upper_bound - lower_bound) * (round_num / total_rounds)
        return noise_strength