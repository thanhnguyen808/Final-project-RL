import torch
import torch.optim as optim
import random
from collections import deque
from magent2.environments import battle_v4
from torch_model import QNetwork
import os

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 100000
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TAU = 0.001  # Soft update factor for target network

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment
env = battle_v4.env(map_size=45, render_mode=None)
obs_shape = env.observation_space("blue_0").shape
action_size = env.action_space("blue_0").n

# Initialize Q-networks and target networks for blue and red
blue_q_network = QNetwork(obs_shape, action_size).to(device)
blue_target_network = QNetwork(obs_shape, action_size).to(device)
red_q_network = QNetwork(obs_shape, action_size).to(device)
red_target_network = QNetwork(obs_shape, action_size).to(device)

# Load checkpoints for blue and red
def load_checkpoint(agent_name, q_network, target_network, default_epsilon):
    if os.path.exists(f"{agent_name}_checkpoint.pth"):
        checkpoint = torch.load(f"{agent_name}_checkpoint.pth", map_location=device)
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        target_network.load_state_dict(checkpoint['target_network_state_dict'])
        epsilon = checkpoint.get('epsilon', default_epsilon)  # Use checkpoint epsilon if available
        print("used checkpoint")
        print(f"{agent_name.capitalize()} checkpoint loaded with epsilon={epsilon:.4f}.")
    elif os.path.exists(f"{agent_name}.pt"):
        q_network.load_state_dict(torch.load(f"{agent_name}.pt", map_location=device))
        target_network.load_state_dict(torch.load(f"{agent_name}.pt", map_location=device))
        epsilon = default_epsilon  # No epsilon in pre-trained weights
        print(f"{agent_name.capitalize()} initialized from {agent_name}.pt with default epsilon={epsilon:.4f}.")
    else:
        print(f"No checkpoint or pretrained weights found for {agent_name}. Starting fresh.")
        epsilon = default_epsilon
    return epsilon

# Initialize epsilon for blue and red
epsilon_blue = load_checkpoint("blue", blue_q_network, blue_target_network, EPSILON_START)
epsilon_red = load_checkpoint("red", red_q_network, red_target_network, EPSILON_START)


# Optimizers
blue_optimizer = optim.Adam(blue_q_network.parameters(), lr=LR)
red_optimizer = optim.Adam(red_q_network.parameters(), lr=LR)

# Replay buffers
blue_replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
red_replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Policy
def epsilon_greedy_policy(state, epsilon, q_network):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    with torch.no_grad():
        state = state.to(device)
        q_values = q_network(state)
        return torch.argmax(q_values).item()

# Training function
def train(q_network, target_network, replay_buffer, optimizer):
    if len(replay_buffer) < BATCH_SIZE:
        return 0.0
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        q_values_next = q_network(next_states)
        max_q_values_next = target_network(next_states).gather(
            1, torch.argmax(q_values_next, dim=1).unsqueeze(1)).squeeze()
        target_q_values = rewards + GAMMA * max_q_values_next * (1 - dones)
    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Soft update
def soft_update(target, source, tau=TAU):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Training loop
num_epochs = 1000
switch_interval = 5  # Number of matches per setup
epsilon = EPSILON_START
for epoch in range(num_epochs):
    env.reset()
    total_loss_blue = 0.0
    total_loss_red = 0.0
    loss_count_blue = 0
    loss_count_red = 0

    # Determine current matchup
    if (epoch // switch_interval) % 3 == 0:
        blue_opponent = red_q_network
        red_opponent = blue_q_network
    elif (epoch // switch_interval) % 3 == 1:
        blue_opponent = blue_q_network
        red_opponent = blue_q_network
    else:
        blue_opponent = red_q_network
        red_opponent = red_q_network

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        done = termination or truncation
        if done:
            action = None
        else:
            observation_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            if agent.startswith("blue"):
                action = epsilon_greedy_policy(observation_tensor, epsilon_blue, blue_q_network)
                if action is not None:
                    blue_replay_buffer.append((observation, action, reward, observation, done))
            else:
                action = epsilon_greedy_policy(observation_tensor, epsilon_red, red_q_network)
                if action is not None:
                    red_replay_buffer.append((observation, action, reward, observation, done))
        env.step(action)
    # Decay epsilon for each team
    epsilon_blue = max(EPSILON_MIN, epsilon_blue * EPSILON_DECAY)
    epsilon_red = max(EPSILON_MIN, epsilon_red * EPSILON_DECAY)
    if len(blue_replay_buffer) > BATCH_SIZE:
        loss_blue = train(blue_q_network, blue_target_network, blue_replay_buffer, blue_optimizer)
        total_loss_blue += loss_blue
        loss_count_blue += 1

    if len(red_replay_buffer) > BATCH_SIZE:
        loss_red = train(red_q_network, red_target_network, red_replay_buffer, red_optimizer)
        total_loss_red += loss_red
        loss_count_red += 1

    soft_update(blue_target_network, blue_q_network)
    soft_update(red_target_network, red_q_network)

    # Logging
    if (epoch + 1) % switch_interval == 0:
        avg_loss_blue = total_loss_blue / loss_count_blue if loss_count_blue > 0 else 0.0
        avg_loss_red = total_loss_red / loss_count_red if loss_count_red > 0 else 0.0
        print(f"Epoch {epoch + 1}: Avg Loss Blue = {avg_loss_blue:.4f}, Avg Loss Red = {avg_loss_red:.4f}, "
              f"Epsilon Blue = {epsilon_blue:.4f}, Epsilon Red = {epsilon_red:.4f}")
    # Save checkpoints periodically
    if (epoch + 1) % 20 == 0:
        torch.save({'q_network_state_dict': blue_q_network.state_dict(),
                    'target_network_state_dict': blue_target_network.state_dict(),
                    'epsilon': epsilon_blue},  # Save current epsilon for blue
                   "blue_checkpoint.pth")
        torch.save({'q_network_state_dict': red_q_network.state_dict(),
                    'target_network_state_dict': red_target_network.state_dict(),
                    'epsilon': epsilon_red},  # Save current epsilon for red
                   "red_checkpoint.pth")
        print(f"Checkpoint saved at epoch {epoch + 1}.")
        # Save additional .pt models
        torch.save(blue_q_network.state_dict(), "blue.pt")
        torch.save(red_q_network.state_dict(), "red.pt")
        print(f"Checkpoint and .pt models saved at epoch {epoch + 1}.")

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    print(f"Epoch {epoch + 1}/1000, Epsilon: {epsilon:.4f}")
env.close()
