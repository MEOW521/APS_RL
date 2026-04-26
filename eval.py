import os
import numpy as np
import pandas as pd
import torch

from aps_rl.env.scheduling_env import SchedulingEnv
from aps_rl.agent.ppo_lagrangian_agent import PPOlagrangianAgent

def eval(data_path, agent=None):
    config_path = "config/config.json"
    env = SchedulingEnv(data_path=data_path, config_path=config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if agent is None:
        ckpt = torch.load("saved_models/best_ppo_lag_model.pth", map_location=device)
        state_dict, train_max_actions = ckpt['model_state_dict'], ckpt['max_actions']
        if train_max_actions < env.action_space.n:
            raise ValueError(f"Train max actions {train_max_actions} is less than env action space {env.action_space.n}")
        agent = PPOlagrangianAgent(max_actions=train_max_actions, env=env, device=device)
        agent.model.load_state_dict(state_dict)
    else:
        train_max_actions = agent.max_actions
    agent.model.eval()

    obs, _ = env.reset()
    done = False
    schedule_sequence = []
    total_reward = 0
    total_costs = np.zeros(5)

    while not done:
        num_actions = env.action_space.n
        pad_len = train_max_actions - num_actions

        if pad_len > 0:
            pad_feats = np.zeros((pad_len, 9), dtype=np.float32)
            padded_candidate_features = np.concatenate([obs["candidate_features"], pad_feats], axis=0)

            pad_mask = np.zeros(pad_len, dtype=np.float32)
            padded_action_mask = np.concatenate([obs['action_mask'], pad_mask], axis=0)

            padded_obs = {
                "global_state": obs["global_state"],
                "candidate_features": padded_candidate_features,
                "action_mask": padded_action_mask
            }
        else:
            padded_obs = obs

        action, _, _, _ = agent.select_action(padded_obs, evaluation=True)
        next_obs, reward, done, truncated, info = env.step(action)
        costs, release_qty = info['cost'], info['release_qty']

        total_reward += reward
        total_costs += costs

        row_info = env.df_agg.iloc[action]

        schedule_sequence.append({
            "生产序号": len(schedule_sequence) + 1,
            "车型": row_info['车型'],
            "白车身": row_info['白车身'],
            "整车物料号": row_info['整车物料号'],
            "生产数量": release_qty
        })

        obs = next_obs
    
    result_df = pd.DataFrame(schedule_sequence)
    result_df.to_csv("data/evaluation/best_ppo_lag_aps_result.csv", index=False, encoding='utf-8')
    print("-"*50)
    print(f"Finish Evaluation! Total Reward {total_reward:.2f} | Total Violations {np.sum(total_costs):.2f}")