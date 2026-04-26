import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from aps_rl.env.scheduling_env import SchedulingEnv
from aps_rl.agent.ppo_lagrangian_agent import PPOlagrangianAgent
from eval import eval


def train():
    data_path = "data/train_data.csv"
    config_path = "config/config.json"
    env = SchedulingEnv(data_path=data_path, config_path=config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOlagrangianAgent(max_actions=env.action_space.n, env=env, device=device)

    worker_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(worker_dir, "runs", "PPO_LAG", current_time))

    num_episodes = 3000
    update_every_n_episodes = 4
    print_every_n_episodes = 10
    eval_every_n_episodes = 50

    best_reward = -np.inf
    best_violations = np.inf

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data/evaluation", exist_ok=True)

    episode_rewards = []
    episode_costs_sums = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_costs = np.zeros(5)
        done = False
        step = 0

        while not done:
            action, log_prob, val_r, val_c = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            costs = info['cost']
            step += 1
            # print(f"Episode {episode} | Step {step} | Action: {action} | Reward: {reward} | Cost: {costs} | Done: {done}")

            agent.store_transition((obs, action, log_prob, reward, costs, val_r, val_c, done))
            obs = next_obs
            episode_reward += reward
            episode_costs += costs

        episode_rewards.append(episode_reward)
        episode_costs_sums.append(episode_costs)

        writer.add_scalar("Performance/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Performance/Total_violations", np.sum(episode_costs), episode)

        if episode % update_every_n_episodes == 0:
            metrics = agent.update()
            avg_reward = np.mean(episode_rewards)
            avg_costs = np.mean(episode_costs_sums, axis=0)

            writer.add_scalar("Loss/Actor", metrics["loss_actor"], episode)
            writer.add_scalar("Loss/Critic", metrics["loss_critic"], episode)
            writer.add_scalar("Lagrangian_Lambda/Model_QTY", metrics["lambdas"][0], episode)
            writer.add_scalar("Lagrangian_Lambda/Biw_QTY", metrics["lambdas"][1], episode)
            writer.add_scalar("Lagrangian_Lambda/Mat_QTY", metrics["lambdas"][2], episode)
            writer.add_scalar("Lagrangian_Lambda/Model_Biw_Types", metrics["lambdas"][3], episode)
            writer.add_scalar("Lagrangian_Lambda/Biw_Mat_Types", metrics["lambdas"][4], episode)

            total_violations = np.sum(avg_costs)

            if total_violations < best_violations or (total_violations == best_violations and avg_reward > best_reward):
                best_reward = avg_reward
                best_violations = total_violations
                torch.save({
                    "model_state_dict":agent.model.state_dict(), 
                    "max_actions": env.action_space.n,
                    },"saved_models/best_ppo_lag_model.pth")
                print(f"Episode {episode} | New best reward: {best_reward:.2f} | New best violations: {best_violations:.2f}")

            episode_rewards = []
            episode_costs_sums = []
        

        if episode % print_every_n_episodes == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Violation: {np.sum(episode_costs):.2f} | Lambdas: {np.round(metrics['lambdas'], 2)}")

        if episode % eval_every_n_episodes == 0:
            print(f"Episode {episode} | Evaluating...")
            eval(data_path="data/eval_data.csv", agent=agent)

    writer.close()
    print("PPO-Lag finished!")

if __name__ == "__main__":
    train()