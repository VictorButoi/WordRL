from typing import Tuple, Any

import numpy as np
import torch
from torch import nn
import gym

from wordle import wordle
from dqn.experience import SequenceReplay, Experience
from wordle.state import WordleState


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

    def get_action(self, state: WordleState, epsilon: float, device: str) -> int:
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


class EmbeddingChars(nn.Module):
    def __init__(self, obs_size, agent_config):
        """
        Args:
            obs_size: observation/state size of the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        emb_size = 8
        self.embedding_layer = nn.Embedding(obs_size, emb_size)
        self.f0 = nn.Sequential(
            nn.Linear(obs_size*emb_size, agent_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(agent_config["hidden_size"], agent_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(agent_config["hidden_size"], word_width),
        )
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

    def forward(self, x):
        emb = self.embedding_layer(x.int())
        y = self.f0(emb.view(x.shape[0], x.shape[1]*self.embedding_layer.embedding_dim))
        z = torch.tensordot(y, self.words.to(self.get_device(x)), dims=[(1,), (0,)])
        return nn.Softmax(dim=1)(z)


class SumChars(nn.Module):
    def __init__(self, obs_size, agent_config):
        """
        Args:
            obs_size: observation/state size of the environment
            hidden_size: size of hidden layers
        """
        #, word_list: List[str], hidden_size: int = 256
        super().__init__()
        word_width = 26*5
        self.f0 = nn.Sequential(
            nn.Linear(obs_size, agent_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(agent_config["hidden_size"], agent_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(agent_config["hidden_size"], word_width),
        )
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

    def forward(self, x):
        y = self.f0(x.float())
        return torch.tensordot(y, self.words.to(self.get_device(x)), dims=((1,), (0,)))

def get_net(obs_size, agent_config):
    if agent_config["type"] == "SumChars":
        return SumChars(obs_size, agent_config)
    elif agent_config["type"] == "EmbeddingChars":
        return EmbeddingChars(obs_size, agent_config)
    else:
        raise ValueError("Network not configured!")
