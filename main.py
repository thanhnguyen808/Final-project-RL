from magent2.environments import battle_v4
import os
import cv2
import torch
import imageio
from torch_model import QNetwork


def save_video_and_gif(frames, video_path, gif_path, fps=35):
    # Save video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    
    # Save gif
    with imageio.get_writer(gif_path, mode='I', duration=1/fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_episode(env, blue_policy, red_policy, video_path, gif_path, max_steps=15000):
    frames = []
    env.reset()
    step = 0

    # Bộ đếm tác tử còn lại của mỗi đội
    blue_agents = {agent for agent in env.agents if "blue" in agent}
    red_agents = {agent for agent in env.agents if "red" in agent}

    while step < max_steps and blue_agents and red_agents:  # Dừng nếu một đội không còn tác tử
        step += 1
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                action = None
                # Loại bỏ agent khỏi đội tương ứng khi "done"
                if agent in blue_agents:
                    blue_agents.remove(agent)
                elif agent in red_agents:
                    red_agents.remove(agent)
            else:
                if agent.startswith("blue"):
                    action = blue_policy(observation)
                elif agent.startswith("red"):
                    action = red_policy(observation)
                else:
                    action = env.action_space(agent).sample()

            env.step(action)

            # Ghi lại khung hình
            if agent == "blue_0" or agent == "red_0":
                frames.append(env.render())

    # Lưu video và gif sau khi trận đấu kết thúc
    save_video_and_gif(frames, video_path, gif_path)


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35

    # Load pretrained model for blue and red agents
    q_network_blue = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
    q_network_blue.load_state_dict(torch.load("blue.pt", map_location="cpu"))
    q_network_red = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n)
    q_network_red.load_state_dict(torch.load("red.pt", map_location="cpu"))

    # Define policies
    def blue_pretrained_policy(observation):
        observation_tensor = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network_blue(observation_tensor)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action

    def red_random_policy(observation):
        return env.action_space("red_0").sample()

    def red_pretrained_policy(observation):
        observation_tensor = torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network_red(observation_tensor)
        action = torch.argmax(q_values, dim=1).numpy()[0]
        return action

    # Maximum steps
    max_steps = 15000  # Max steps in each match

    # Run Blue vs Random
    print("Running Blue vs Random...")
    run_episode(env, blue_pretrained_policy, red_random_policy, 
                f"{vid_dir}/blue_vs_random.mp4", 
                f"{vid_dir}/blue_vs_random.gif", max_steps)
    print("Done with Blue vs Random")

    # Run Blue vs Red
    print("Running Blue vs Red...")
    run_episode(env, blue_pretrained_policy, red_pretrained_policy, 
                f"{vid_dir}/blue_vs_red.mp4", 
                f"{vid_dir}/blue_vs_red.gif", max_steps)
    print("Done with Blue vs Red")

    env.close()
