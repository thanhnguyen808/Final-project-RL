import torch
import torch.optim as optim
import random
from collections import deque
from magent2.environments import battle_v4
from torch_model import QNetwork
import os

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128  # Tăng batch size
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

# Initialize Q-network and target network for blue
q_network = QNetwork(obs_shape, action_size).to(device)
target_network = QNetwork(obs_shape, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LR)

# Kiểm tra và tải mô hình red từ red.pt
if os.path.exists("red.pt"):
    red_network = QNetwork(obs_shape, action_size).to(device)
    red_network.load_state_dict(torch.load("red.pt", map_location=device))  # Load red model to device
    print("Red model loaded successfully.")
else:
    print("No red model found. Please check if red.pt exists.")