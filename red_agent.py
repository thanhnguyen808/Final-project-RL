import torch
from torch_model import QNetwork

class RedAgent:
    def __init__(self, obs_shape, action_size, model_path="red.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(obs_shape, action_size).to(self.device)
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_network.eval()

    def select_action(self, observation):
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return torch.argmax(q_values).item()
