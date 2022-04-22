import torch
import torch.nn as nn
import gym


class WorDQN():
    """Basic DQN Model."""

    def __init__(
            self,
            initialize_winning_replays: str = None,
            deep_q_network: str = 'SumChars',
            batch_size: int = 1024,
            lr: float = 1e-4,
            env: str = 'Wordle-v2-10-visualized',
            gamma: float = 0.9,
            sync_rate: int = 10,
            replay_size: int = 1000,
            hidden_size: int = 256,
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
        # currently broken, need to actually pass stuff
        #changed from gym.make(self.hparams.env)
        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.winning_steps = 0
        self.wins = 0
        self.losses = 0
        self.rewards = 0

        self.policy_network = construct(deep_q_network,
                                        obs_size=obs_size,
                                        n_actions=n_actions, hidden_size=hidden_size,
                                        word_list=self.env.words)
        self.value_network = construct(deep_q_network,
                                       obs_size=obs_size,
                                       n_actions=n_actions, hidden_size=hidden_size,
                                       word_list=self.env.words)

        self.dataset = RLDataset(winners=SequenceReplay(
                                 self.hparams.replay_size//2,
                                 self.hparams.initialize_winning_replays),
                                 losers=SequenceReplay(
                                     self.hparams.replay_size//2),
                                 sample_size=self.hparams.episode_length)

        self.agent = Agent(self.policy_network, self.env.action_space)
        self.state = self.env.reset()
        self.total_reward = 0
        self.episode_reward = 0
        self.total_games_played = 0
        self.populate(warm_start_steps)

    def populate(self, steps: int = 1000):
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.play_game(epsilon=1.)

    def forward(self, x):
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.policy_network(x)
        return output

    def dqn_mse_loss(self, batch):
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        loss_fn = nn.MSELoss()
        states, actions, rewards, dones, next_states = batch
        # actions used to be used here
        state_action_values = self.net(states)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return loss_fn(state_action_values, expected_state_action_values)

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

        winning_steps = self.env.max_turns - \
            wordle.state.remaining_steps(self.state)
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
        exp = Experience(self.state.copy(), action, reward,
                         done, new_state.copy(), self.env.goal_word)

        self.state = new_state
        return exp


def construct(network,
              obs_size,
              n_actions,
              hidden_size,
              word_list):
    pass
