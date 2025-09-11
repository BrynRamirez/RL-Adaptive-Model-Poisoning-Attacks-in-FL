import torch
import copy
import numpy as np

class AdaptiveModelPoisoning():
    def __init__(self, strategy='scale',max_scale=10.0, noise_std=0.1):
        self.strategy = strategy
        self.max_scale = max_scale
        self.noise_std = noise_std

    def poison(self, global_model, client_model, round_num, total_rounds):
        poisoned_model = copy.deepcopy(client_model)

        if self.strategy == 'scale':
            scale_factor = self._get_scale_factor(round_num, total_rounds)
            for param_global, param_client in zip(global_model.parameters(), poisoned_model.parameters()):
                update = param_client.data - param_global.data
                param_client.data = param_global.data + scale_factor * update

        elif self.strategy == 'noise':
            noise_std = self._adaptive_noise(round_num, total_rounds)
            for param in poisoned_model.parameters():
                noise = torch.normal(mean=0.0, std=noise_std, size=param.data.size(), device=param.data.device)
                param.data += noise

        return poisoned_model

    def _get_scale_factor(self, round_num, total_rounds):
        """Adapt the scale factor as rounds progress"""
        return 1.0 + (self.max_scale - 1.0) * (round_num / total_rounds)

    def _adaptive_noise(self, round_num, total_rounds):
        """Generate adaptive noise"""
        noise_strength = 0.1 + 2.0 * (round_num / total_rounds) # 0.1 round 0, 2.1 final of round
        return noise_strength