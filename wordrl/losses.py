import torch.nn as nn


def dqn_mse_loss(batch, gamma, net, target_net):
    states, actions, rewards, dones, next_states = batch
    state_action_values = net(states).gather(1,
                                             actions.unsqueeze(1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values
    * game + rewards

    return nn.MSE()(state_action_values, expected_state_action_values)

