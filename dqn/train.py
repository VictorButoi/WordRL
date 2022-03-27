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

    if self.global_step % 100 == 0:
        if len(self.dataset.winners) > 0:
            winner = self.dataset.winners.buffer[-1]
            game = f"goal: {self.env.words[winner[0].goal_id]}\n"
            for i, xp in enumerate(winner):
                game += f"{i}: {self.env.words[xp.action]}\n"
            self.writer.add_text("game sample/winner", game,
                                 global_step=self.global_step)
        if len(self.dataset.losers) > 0:
            loser = self.dataset.losers.buffer[-1]
            game = f"goal: {self.env.words[loser[0].goal_id]}\n"
            for i, xp in enumerate(loser):
                game += f"{i}: {self.env.words[xp.action]}\n"
            self.writer.add_text("game sample/loser", game,
                                 global_step=self.global_step)
        self.writer.add_scalar(
            "train_loss", loss, global_step=self.global_step)
        self.writer.add_scalar(
            "total_games_played", self.total_games_played, global_step=self.global_step)

        self.writer.add_scalar("winner_buffer", len(
            self.dataset.winners), global_step=self.global_step)
        self.writer.add_scalar("loser_buffer", len(
            self.dataset.losers), global_step=self.global_step)

        self.writer.add_scalar(
            "lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
        self.writer.add_scalar(
            "wins", self._wins, global_step=self.global_step)
        self.writer.add_scalar("reward_per_game", self._rewards /
                               (self._wins+self._losses), global_step=self.global_step)
        if self._wins > 0:
            self.writer.add_scalar(
                "avg_winning_turns", self._winning_steps/self._wins, global_step=self.global_step)
        self._winning_steps = 0
        self._wins = 0
        self._losses = 0
        self._rewards = 0

    return OrderedDict({"loss": loss, "log": log, "progress_bar": status})
