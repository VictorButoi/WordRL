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

@torch.no_grad()
def play_one_game(agent, state, env, dataset, epsilon, device):
    done = False
    cur_seq = list()
    reward = 0
    while not done:
        action = agent.get_action(state, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = env.step(action)
        exp = wdl.experience.Experience(state.copy(), action, reward, done, new_state.copy(), env.goal_word)
        state = new_state
        done = exp.done
        reward = exp.reward
        cur_seq.append(exp)

    winning_steps = env.max_turns - state[0]
    print(winning_steps)
    if reward > 0:
        dataset.winners.append(cur_seq)
    else:
        dataset.losers.append(cur_seq)
    
    state = env.reset()

    return state, reward, winning_steps
    

def run_dqn_experiment(config):
    env = gym.make('Wordle-v2-10-visualized')
    # ndarray
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

    dataset = wdl.experience.RLDataset(winners=wdl.experience.SequenceReplay(config["dataset"]), 
                                       losers=wdl.experience.SequenceReplay(config["dataset"]),
                                       sample_size=config["dataset"]["eps_length"])

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config["training"]["batch_size"])
    optimizer = torch.optim.Adam(net.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    # high level statistics
    total_reward = 0
    episode_reward = 0
    total_games_played = 0
    
    # global step tracks number of optimizer steps
    global_step = 0
    
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
            state, _, _ = play_one_game(agent, state, env, dataset, epsilon, device)
        else:
            # Training step
            for batch in dataloader:
                epsilon = max(config["training"]["eps_end"],
                          config["training"]["eps_state"] - total_games_played /
                          config["training"]["eps_last_frame"])
                # step through environment with agent
                with torch.no_grad():
                    state, reward, winning_steps = play_one_game(agent, state, env, dataset, epsilon, device)
                    
                total_games_played += 1
 
                if reward > 0:
                    wins += 1
                    winning_steps += winning_steps
                else:
                    losses += 1
                rewards += reward

                # standard pytorch training loop
                optimizer.zero_grad()
                loss = wdl.losses.dqn_mse_loss(batch, config["training"]["gamma"], net, target_net)
                loss.backward()
                optimizer.step()

                global_step += 1

                        # Soft update of target network
                if global_step % config["experiment"]["sync_rate"] == 0:
                    target_net.load_state_dict(net.state_dict())

                log = {
                    "total_reward": torch.tensor(total_reward).to(device),
                    "reward": torch.tensor(reward).to(device),
                    "train_loss": loss.detach(),
                }
                status = {
                    "steps": torch.tensor(global_step).to(device),
                    "total_reward": torch.tensor(total_reward).to(device),
                }

                if global_step % config["experiment"]["steps_per_update"] == 0:
                    if len(dataset.winners) > 0:
                        winner = dataset.winners.buffer[-1]
                        game = f"goal: {winner[0].goal_id}\n"
                        for i, xp in enumerate(winner):
                            game += f"{i}: {env.words[xp.action]}\n"
                    if len(dataset.losers) > 0:
                        loser = dataset.losers.buffer[-1]
                        game = f"goal: {loser[0].goal_id}\n"
                        for i, xp in enumerate(loser):
                            game += f"{i}: {env.words[xp.action]}\n"

                    winning_steps = 0
                    wins = 0
                    losses = 0
                    rewards = 0

