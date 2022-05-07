from typing import Tuple, Any

import numpy as np
import torch
from torch import nn
import gym

import wordrl as wdl
from .experience import SequenceReplay, Experience


class Agent:
    def __init__(self,
                 net: nn.Module,
                 action_space: Any):
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.net = net
        self.action_space = action_space

    def get_action(self, state: wdl.wordle.state.WordleState, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            state = torch.tensor([state]).to(device)
            q_values = self.net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action
