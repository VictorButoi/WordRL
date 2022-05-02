# wordrl imports
import wordrl as wdl

# torch imports
import torch

# misc imports
import gym
import os
import numpy as np


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
                        for x in open('tmp', 'r').readlines()]
    if len(memory_available) == 0:
        return 0
    else:
        return np.argmax(memory_available)


def run_experiment(config):
    env = gym.make('Wordle-v2-10-visualized')
    state = env.reset()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    num_eps = config["experiment"]["num_episodes"]
    if config["experiment"]["use_gpu"]:
        device = get_freer_gpu()
    else:
        device = torch.device("cpu")

    net = wdl.agent.get_net(obs_size, n_actions, config["agent"])
    target_net = wdl.agent.get_net(obs_size, n_actions, config["agent"])
    agent = wdl.agent.Agent(net, env.action_space)

    dataset = wdl.experience.RLDataset(winners=wdl.experience.SequenceReplay(
        config["dataset"]["replay_size"]//2, config["dataset"]["init_winning_replays"]), losers=wdl.experience.SequenceReplay(config["dataset"]["replay_size"]//2))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config["training"]["batch_size"])
    optimizer = torch.optim.Adam(net.parameters(
    ), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    # high level statistics
    total_reward = 0
    episode_reward = 0
    total_games_played = 0

    # low level statistics
    wins = 0
    losses = 0
    winning_steps = 0
    rewards = 0

    #
    # Kind of tricky scheme: Episode = Game
    #
    # Looping to play multiple games of Wordrl
    for i in range(num_eps + config["training"]["warmup"]):

        # provide small warmup period
        if i < config["training"]["warmup"]:
            epsilon = 1
        else:
            epsilon = max(config["training"]["eps_end"],
                          config["training"]["eps_state"] - total_games_played /
                          config["training"]["eps_last_frame"])

        # Training step
        for batch in dataloader:
            state, action, reward, done, new_state = batch

            # Play one game of WordRL!
            with torch.no_grad():

                done = False
                cur_seq = list()
                reward = 0

                # Get one action at a time
                while not done:

                    # take your action
                    action = agent.get_action(state, epsilon, device)
                    # take a step in your environment
                    new_state, reward, done, _ = env.step(action)
                    # make an experience object out of it
                    exp = wdl.experience.Experience(state.copy(), action,
                                                    reward, new_state.copy(), env.goal_word)
                    # set the new state
                    state = new_state

                done = exp.done
                reward = exp.reward
                cur_seq.append(exp)

                winning_steps = env.max_turns - state[0]

                # build up our experience dataset
                if reward > 0:
                    dataset.winners.append(cur_seq)
                else:
                    dataset.losers.append(cur_seq)

                # start the game over
                state = env.reset()

            # one more game has been played, updated reward
            total_games_played += 1
            rewards += reward

            loss = wdl.losses.dqn_mse_loss(
                batch, config["training"]["gamma"], net, target_net)

            # keep track of wins and losses
            if reward > 0:
                wins += 1
                winning_step += winning_steps
            else:
                losses += 1

            # If it is time to sync the old value with a new one
            if total_games_played % config["experiment"]["sync_rate"] == 0:
                target_net.load_state_dict(net.state_dict())

            if total_games_played % config["experiment"]["steps_per_update"] == 0:
                if len(dataset.winners) > 0:
                    winner = dataset.winners.buffer[-1]
                    game = f"goal: {env.words[winner[0].goal_id]}\n"
                    # Go through the LAST winners experience
                    for i, xp in enumerate(winner):
                        game += f"{i}: {env.words[xp.action]}\n"
                if len(dataset.losers) > 0:
                    loser = dataset.losers.buffer[-1]
                    game = f"goal: {env.words[loser[0].goal_id]}\n"
                    for i, xp in enumerate(loser):
                        game += f"{i}: {env.words[xp.action]}\n"
                winning_steps = 0
                wins = 0
                losses = 0
                rewards = 0
                exp = wdl.experience.Experience(state.copy(), action, reward,
                                                new_state.copy(), env.goal_word)

            winning_steps = env.max_turns - state[0]

            if reward > 0:
                wins += 1
                winning_steps += winning_steps
            else:
                losses += 1

            # TODO: define what reward is

        # TODO: define what reward is
        rewards += reward
        total_games_played += 1

        loss = wdl.losses.dqn_mse_loss(batch)

        # If it is time to sync the old value with a new one
        if total_games_played % config["experiment"]["sync_rate"] == 0:
            # TODO: define a target_network and a net
            target_net.load_state_dict(net.state_dict())

        if total_games_played % config["experiment"]["steps_per_update"] == 0:
            if len(dataset.winners) > 0:
                winner = dataset.winners.buffer[-1]
                game = f"goal: {env.words[winner[0].goal_id]}\n"
                # Go through the LAST winners experience
                for i, xp in enumerate(winner):
                    game += f"{i}: {env.words[xp.action]}\n"
            if len(dataset.losers) > 0:
                loser = dataset.losers.buffer[-1]
                game = f"goal: {env.words[loser[0].goal_id]}\n"
                for i, xp in enumerate(loser):
                    game += f"{i}: {env.words[xp.action]}\n"
            winning_steps = 0
            wins = 0
            losses = 0
            rewards = 0
