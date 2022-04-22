#TODO: import env
import * from dqn_utils

num_eps = config["experiment"]["num_episodes"]
#TODO: get the device of the current batch
device = get_device(batch)
total_games_played = 0


#Record wins and losses, clear these at some point

wins = 0
losses = 0
winning_steps = 0

#Record reward
rewards = 0

losses

#TODO: do some kind of looping to play multiple games
for i in range(num_episodes):
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
            

            exp = Experience(state.copy(), action, reward, new_state.copy(), env.goal_word)
        
        #TODO: define WordRL object
        winning_steps = env.max_turns - wordle.state.remaining_steps(state)

        if reward > 0:
            dataset.winners.append(sequence)
            wins += 1
            winning_steps += winning_steps
        else:
            dataset.losers.append(sequence)
            losses += 1
        
        #TODO: define what reward is 
        
        total_games_played += 1


