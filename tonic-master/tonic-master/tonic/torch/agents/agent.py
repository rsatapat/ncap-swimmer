import os
import random

import numpy as np
import torch

from tonic import agents, logger  # noqa


class Agent(agents.Agent):
    def initialize(self, observation_space, action_space, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        self.observation_space = observation_space
        self.action_space = action_space

        # Cascade space specs down to PyTorch networks (creates MLP layers)
        if hasattr(self, 'model') and self.model is not None:
            self.model.initialize(observation_space, action_space)

        # Cascade action spec down to exploration noise (sets action_size)
        # if hasattr(self, 'exploration') and self.exploration is not None:
        #     self.exploration.initialize(action_space)
        

    def save(self, path):
        path = path + '.pt'
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        path = path + '.pt'
        logger.log(f'\nLoading weights from {path}')
        self.model.load_state_dict(torch.load(path))
