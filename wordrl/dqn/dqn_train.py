from dataclasses import dataclass
from typing import Tuple, List

from collections import OrderedDict

import fire
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

import wordrl as wdl

import wandb
import gym

AVAIL_GPUS = min(1, torch.cuda.device_count())

# Handle things that used to get handled in the import
from .agent import Agent
from .experience import SequenceReplay, RLDataset, Experience
from .embeddingchars import EmbeddingChars
from .mlp import MLP
from .sumchars import SumChars

_registry = {}

def register(ctor, name):
    _registry[name] = ctor

def construct(name, **kwargs):
    return _registry[name](**kwargs)

register(MLP, "MLP")
register(EmbeddingChars, "EmbeddingChars")
register(SumChars, "SumChars")


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            initialize_winning_replays: str = None,
            deep_q_network: str = 'SumChars',
            batch_size: int = 1024,
            lr: float = 1e-2,
            weight_decay: float = 1.e-4,
            env: str = "Wordle-v2-10-visualized",
            gamma: float = 0.9,
            sync_rate: int = 10,
            replay_size: int = 1000,
            hidden_size: int = 256,
            num_workers: int = 0,
            warm_start_size: int = 1000,
            eps_last_frame: int = 10000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 25,
            warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self._winning_steps = 0
        self._wins = 0
        self._losses = 0
        self._rewards = 0

        print("dqn:", self.env.spec.id, self.env.spec.max_episode_steps, n_actions, obs_size)

        self.net = construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, word_list=self.env.words)
        self.target_net = construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, word_list=self.env.words)

        self.dataset = RLDataset(
            winners=SequenceReplay(self.hparams.replay_size//2, self.hparams.initialize_winning_replays),
            losers=SequenceReplay(self.hparams.replay_size//2),
            sample_size=self.hparams.episode_length)

        self.agent = Agent(self.net, self.env.action_space)
        self.state = self.env.reset()
        self.total_reward = 0
        self.episode_reward = 0
        self.total_games_played = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.play_game(epsilon=1., device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    @torch.no_grad()
    def play_game(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool]:
        done = False
        cur_seq = list()
        reward = 0
        while not done:
            exp = self.play_step(epsilon, device)
            done = exp.done
            reward = exp.reward
            cur_seq.append(exp)

        winning_steps = self.env.max_turns - wdl.wordle.state.remaining_steps(self.state)
        if reward > 0:
            self.dataset.winners.append(cur_seq)
        else:
            self.dataset.losers.append(cur_seq)
        self.state = self.env.reset()

        return reward, winning_steps

    def play_step(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Experience:
        action = self.agent.get_action(self.state, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state.copy(), action, reward, done, new_state.copy(), self.env.goal_word)

        self.state = new_state
        return exp

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
            )

        # step through environment with agent
        with torch.no_grad():
            reward, winning_steps = self.play_game(epsilon, device)
        self.total_games_played += 1
        if reward > 0:
            self._wins += 1
            self._winning_steps += winning_steps
        else:
            self._losses += 1

        self._rewards += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss.detach(),
        }
               
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if self.global_step % 50 == 0:
            
            metrics = {
                "train_loss": loss.detach(),
                "total_games_played": self.total_games_played,
                "lose_ratio": self._losses/(self._wins+self._losses),
                "wins": self._wins,
                "reward_per_game": self.total_reward / (self._wins+self._losses),
                "global_step": self.global_step,
            }
            
            wandb.log(metrics)
            
            if len(self.dataset.winners) > 0:
                winner = self.dataset.winners.buffer[-1]
                game = f"goal: {winner[0].goal_id}\n"
                for i, xp in enumerate(winner):
                    game += f"{i}: {self.env.words[xp.action]}\n"
                    
            if len(self.dataset.losers) > 0:
                loser = self.dataset.losers.buffer[-1]
                game = f"goal: {loser[0].goal_id}\n"
                for i, xp in enumerate(loser):
                    game += f"{i}: {self.env.words[xp.action]}\n"
         
            self._winning_steps = 0
            self._wins = 0
            self._losses = 0
            self._rewards = 0

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def main(config):
    with wandb.init(project='wordle-solver'):
        wandb.config.update(config)
        wandb.run.name = config["experiment"]["name"]
        
        model = DQNLightning(
            initialize_winning_replays=config["dataset"]["init_winning_replays"],
            deep_q_network=config["agent"]["network"],
            env=config["dataset"]["env"],
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            replay_size=config["dataset"]["replay_size"],
            batch_size=config["training"]["batch_size"],
            sync_rate=config["experiment"]["sync_rate"],
            episode_length=config["dataset"]["episode_length"],
            hidden_size=config["agent"]["hidden_size"],
            num_workers=config["experiment"]["num_workers"],
            eps_start=config["training"]["max_eps"],
            eps_end=config["training"]["min_eps"],
            eps_last_frame=int(config["training"]["epochs"]*config["training"]["last_frame_cutoff"]),
        )

        @dataclass
        class SaveBufferCallback(Callback):
            buffer: SequenceReplay
            filename: str

            def on_train_end(self, trainer, pl_module):
                path = f'{trainer.log_dir}/checkpoints'
                fname = self.filename,
                self.buffer.save(f'{path}/{fname}')

        save_buffer_callback = SaveBufferCallback(buffer=model.dataset.winners, filename='sequence_buffer.pkl')
        model_checkpoint = ModelCheckpoint(every_n_epochs=config["training"]["checkpoint_every_n_epochs"])
        trainer = Trainer(
            gpus=AVAIL_GPUS,
            max_epochs=config["training"]["epochs"],
            enable_checkpointing=True,
            callbacks=[model_checkpoint, save_buffer_callback],
            resume_from_checkpoint=None
        )

        trainer.fit(model)


def train_func(config):    
    main(config)
