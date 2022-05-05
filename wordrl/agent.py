from typing import Tuple, Any

import numpy as np
import torch
from torch import nn
import gym

from typing import List


from wordrl.experience import SequenceReplay, Experience
from wordrl.envs.wordle_env_v2_visualized import WORDS


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

    def get_action(self, state, epsilon: float, device: str) -> int:
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


class A2CEmbeddingChars(nn.Module):
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
            nn.Linear(agent_config["hidden_size"],
                      agent_config["hidden_size"]),
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
        y = self.f0(emb.view(x.shape[0], x.shape[1]
                    * self.embedding_layer.embedding_dim))
        z = torch.tensordot(y, self.words.to(
            self.get_device(x)), dims=[(1,), (0,)])
        return nn.Softmax(dim=1)(z)
 
class A2CSumChars(nn.Module):
    def __init__(self, obs_size: int, word_list: List[str], n_hidden: int = 1, hidden_size: int = 256):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, word_width))
        layers.append(nn.ReLU())

        self.f0 = nn.Sequential(*layers)
        word_array = np.zeros((word_width, len(word_list)))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[j*26 + (ord(c) - ord('A')), i] = 1
        self.words = torch.Tensor(word_array)

        self.actor_head = nn.Linear(word_width, word_width)
        self.critic_head = nn.Linear(word_width, 1)

    def forward(self, x):
        y = self.f0(x.float())
        a = torch.log_softmax(
            torch.tensordot(self.actor_head(y),
                            self.words.to(self.get_device(y)),
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(y)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index


class ActorCriticAgent:
    """Actor-Critic based agent that returns an action based on the networks policy."""

    def __init__(self, net):
        self.net = net

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logprobs, _ = self.net(torch.tensor([states], device=device))
        probabilities = logprobs.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        # Note that this is much faster than numpy.random.choice
        cdf = np.cumsum(prob_np, axis=1)
        cdf[:, -1] = 1.  # Ensure cumsum adds to 1
        select = np.random.random(cdf.shape[0])
        actions = [
            np.searchsorted(cdf[row, :], select[row])
            for row in range(cdf.shape[0])
        ]

        return actions


class GreedyActorCriticAgent:
    def __init__(self, net):
        self.net = net

    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        logprobs, _ = self.net(torch.tensor([states], device=device))
        probabilities = logprobs.exp().squeeze(dim=-1)
        prob_np = probabilities.data.cpu().numpy()

        actions = np.argmax(prob_np, axis=1)

        return list(actions)
    

class DQNEmbeddingChars(nn.Module):
    def __init__(self,
                 obs_size: int,
                 word_list: List[str],
                 n_hidden: int = 1,
                 hidden_size: int = 256,
                 n_emb: int = 32,
                 ):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        word_width = 26*5
        self.n_emb = n_emb

        layers = [
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(n_hidden):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, self.n_emb))

        self.f_state = nn.Sequential(*layers)

        self.actor_head = nn.Linear(self.n_emb, self.n_emb)
        self.critic_head = nn.Linear(self.n_emb, 1)

        word_array = np.zeros((len(word_list), word_width))
        for i, word in enumerate(word_list):
            for j, c in enumerate(word):
                word_array[i, j*26 + (ord(c) - ord('A'))] = 1
        self.words = torch.Tensor(word_array)

        # W x word_width -> W x emb
        self.f_word = nn.Sequential(
            nn.Linear(word_width, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_emb),
        )

    def forward(self, x):
        fs = self.f_state(x.float())
        fw = self.f_word(
            self.words.to(self.get_device(x)),
        ).transpose(0, 1)

        a = torch.log_softmax(
            torch.tensordot(self.actor_head(fs), fw,
                            dims=((1,), (0,))),
            dim=-1)
        c = self.critic_head(fs)
        return a, c

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index


class DQNSumChars(nn.Module):
    def __init__(self, obs_size, agent_config):
        """
        Args:
            obs_size: observation/state size of the environment
            hidden_size: size of hidden layers
        """
        # , word_list: List[str], hidden_size: int = 256
        super().__init__()
        word_width = 26*5
        word_list = WORDS
        self.f0 = nn.Sequential(
            nn.Linear(obs_size, agent_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(agent_config["hidden_size"],
                      agent_config["hidden_size"]),
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
        return torch.tensordot(y, self.words, dims=((1,), (0,)))


def get_net(obs_size, n_actions, agent_config):
    if agent_config["type"] == "SumChars":
        return SumChars(obs_size, agent_config)
    elif agent_config["type"] == "EmbeddingChars":
        return EmbeddingChars(obs_size, agent_config)
    else:
        raise ValueError("Network not configured!")
