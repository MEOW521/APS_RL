import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from aps_rl.net.ppo_lagrangian_network import PPOLagrangianNetwork


class PPOlagrangianAgent:
    def __init__(self, max_actions: int, env, device='cpu'):
        self.device = device
        self.max_actions = max_actions
        self.state_dim = env.observation_space['global_state'].shape[0]
        self.action_feat_dim = env.observation_space['candidate_features'].shape[1]
        
        self.model = PPOLagrangianNetwork(max_actions, self.state_dim, self.action_feat_dim)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.lambdas = nn.Parameter(torch.zeros(5, dtype=torch.float32, device=device))
        self.lambdas_optimizer = torch.optim.Adam([self.lambdas], lr=5e-3)

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.memory = []

    def select_action(self, obs, evaluation=False):
        global_state = torch.FloatTensor(obs['global_state']).unsqueeze(0).to(self.device)
        candidate_features = torch.FloatTensor(obs['candidate_features']).unsqueeze(0).to(self.device)
        action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model.forward_actor(global_state, candidate_features, action_mask)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if evaluation:
                return torch.argmax(probs, dim=-1).item(), 0, 0, np.zeros(5)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            val_r, val_c = self.model.forward_critic(global_state)

        return action.item(), log_prob.item(), val_r.item(), val_c.squeeze(0).cpu().numpy()
    
    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0: return {}

        global_states_list = []
        candidate_features_list = []
        action_masks_list = []
        actions, log_probs, rewards, costs = [], [], [], []
        vals_r, vals_c, dones = [], [], []

        for obs, action, log_prob, reward, cost, val_r, val_c, done in self.memory:
            global_states_list.append(obs['global_state'])
            candidate_features_list.append(obs['candidate_features'])
            action_masks_list.append(obs['action_mask'])
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            costs.append(cost)
            vals_r.append(val_r)
            vals_c.append(val_c)
            dones.append(done)

        global_states = torch.FloatTensor(np.array(global_states_list)).to(self.device)
        candidate_features = torch.FloatTensor(np.array(candidate_features_list)).to(self.device)
        action_masks = torch.FloatTensor(np.array(action_masks_list)).to(self.device)
        old_actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        costs = torch.FloatTensor(costs).to(self.device)
        vals_r = torch.FloatTensor(vals_r).to(self.device)
        vals_c = torch.FloatTensor(vals_c).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算reward的GAE
        returns_r, adv_r = [], []
        gae_r = 0
        for i in reversed(range(len(rewards))):
            next_vr = 0 if i == len(rewards)-1 else vals_r[i+1]
            delta_r = rewards[i] + self.gamma * next_vr * (1 - dones[i]) - vals_r[i]
            gae_r = delta_r + self.gamma * self.lmbda * (1 - dones[i]) * gae_r
            returns_r.insert(0, gae_r + vals_r[i])
            adv_r.insert(0, gae_r)
        
        # 计算cost的GAE
        returns_c, adv_c = [], []
        gae_c = torch.zeros(5).to(self.device)
        for i in reversed(range(len(costs))):
            next_vc = torch.zeros(5).to(self.device) if i == len(costs)-1 else vals_c[i+1]
            delta_c = costs[i] + self.gamma * next_vc * (1 - dones[i]) - vals_c[i]
            gae_c = delta_c + self.gamma * self.lmbda * (1 - dones[i]) * gae_c
            returns_c.insert(0, gae_c + vals_c[i])
            adv_c.insert(0, gae_c)

        returns_r = torch.stack(returns_r).to(self.device)
        returns_c = torch.stack(returns_c).to(self.device)
        adv_r = torch.stack(adv_r).to(self.device)
        adv_c = torch.stack(adv_c).to(self.device)

        current_lambdas = self.lambdas.detach()
        adv_lag = adv_r - (adv_c * current_lambdas).sum(dim=-1)
        adv_lag = (adv_lag - adv_lag.mean()) / (adv_lag.std() + 1e-8)

        actor_losses = []
        critic_losses = []

        for _ in range(self.k_epochs):
            logits = self.model.forward_actor(global_states, candidate_features, action_masks)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            new_vr, new_vc = self.model.forward_critic(global_states)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * adv_lag
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv_lag
            actor_loss = -torch.min(surr1, surr2).mean()

            loss_vr = nn.SmoothL1Loss()(new_vr.squeeze(-1), returns_r)
            loss_vc = nn.SmoothL1Loss()(new_vc, returns_c)
            critic_loss = loss_vr + loss_vc

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        mean_cost_returns = returns_c.mean(dim=0)
        limits = torch.zeros(5).to(self.device)
        lambda_loss = -(self.lambdas * (mean_cost_returns - limits).detach()).sum()
        self.lambdas_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambdas_optimizer.step()

        with torch.no_grad():
            self.lambdas.clamp_(min=0.0)

        self.memory.clear()

        return {
            "loss_actor": np.mean(actor_losses),
            "loss_critic": np.mean(critic_losses),
            "lambdas": self.lambdas.detach().cpu().numpy(),
            "avg_costs": mean_cost_returns.detach().cpu().numpy(),
        }
