#torch imports
import torch

#misc imports

def get_device(self, batch) -> str:
    """Retrieve device currently being used by minibatch."""
    return batch[0].device.index if self.on_gpu else "cpu"


#TODO: import env

num_eps = config["experiment"]["num_episodes"]
device = get_device(batch)
net = None
target_net = None

#TODO: define an RL Dataset
dataset = RLDataset(winners=SequenceReplay(config["dataset"]["replay_size"]//2, config["dataset"]["init_winning_replays"]),
losers = Sequence_Replay(config["dataset"]["replay_size"]//2,
        sample_size=config["dataset"]["sample_size"]))

dataloader = torch.utils.data.Dataloader(dataset=dataset, batch_size=config["training"]["batch_size"])
#define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
total_games_played = 0

for i in range(num_episodes):

    #TODO: there is something of a batch defined up here
    batch = None

    with torch.no_grad():
        # Initialize the environment and state
        env.reset()
        sequence = ()
        
        #play one game of WordRL!
        while not done:
            #TODO: define epsilon
            #take your action
            action = agent.get_action(state, epsilon, device)
            
            #take a step in your environment
            new_state, reward, done, _ = env.step(action)
            
            #TODO: define an experience object, might be useful
            exp = Experience(state.copy(), action, reward, new_state.copy(), env.goal_word)
        
        #TODO: define WordRL object
        winning_steps = env.max_turns - wordle.state.remaining_steps(state)

    #TODO: define datasets objects, has at least winners and losers, define whatever
    # the fuck wins / losses are
    if reward > 0:
        dataset.winners.append(sequence)
        _wins += 1
        _winning_step += winning_steps
    else:
        dataset.losers.append(sequence)
        _losses += 1
        
    #TODO: define what reward is 
    _rewards += reward
    total_games_played += 1

    #TODO: Make this loss function and some losses.py file get training loss
    loss = wdl.losses.dqn_mse_loss(batch)
    
    #If it is time to sync the old value with a new one
    if global_setep % config["experiment"]["sync_rate"] == 0:
        #TODO: define a target_network and a net
        target_net.load_state_dict(net.state_dict())


    if global_step % config["experiment"]["steps_per_update"] == 0:
        if len(dataset.winners) > 0:
            winner = dataset.winners.buffer[-1]
            game = f"goal: {env.words[winner[0].goal_id]}\n"
            #Go through the LAST winners experience
            for i, xp in enumerate(winner):
                game += f"{i}: {env.words[xp.action]}\n"
        if len(dataset.losers) > 0:
            loser = dataset.losers.buffer[-1]
            game = f"goal: {env.words[loser[0].goal_id]}\n"
            for i, xp in enumerate(loser):
                game += f"{i}: {env.words[xp.action]}\n"
        _winning_steps = 0
        _wins = 0
        _losses = 0
        _rewards = 0




