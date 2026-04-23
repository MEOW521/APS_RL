import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + res)


class PPOLagrangianNetwork(nn.Module):
    def __init__(self, max_actions: int, state_dim: int, action_feat_dim: int):
        super().__init__()
        self.max_actions = max_actions
        self.state_dim = state_dim
        self.action_feat_dim = action_feat_dim
        self.hidden_dim = 256

        # Actor
        self.actor_in = nn.Linear(self.state_dim+self.action_feat_dim, self.hidden_dim)
        self.actor_res = ResidualBlock(self.hidden_dim)
        self.actor_out = nn.Linear(self.hidden_dim, 1)

        # Value Critic
        self.value_in = nn.Linear(self.state_dim, self.hidden_dim)
        self.value_res = ResidualBlock(self.hidden_dim)
        self.value_out = nn.Linear(self.hidden_dim, 1)

        # Cost Critic
        self.cost_in = nn.Linear(self.state_dim, self.hidden_dim)
        self.cost_res = ResidualBlock(self.hidden_dim)
        self.cost_out = nn.Linear(self.hidden_dim, 5)

    def forward_actor(self, global_state, candidate_features, action_mask):
        # actor根据global_state和candidate_features，计算每个动作的logits，同时需要考虑action_mask

        # B: batch size, N: action dim
        B, N, D = candidate_features.shape
        global_state_expanded = global_state.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([global_state_expanded, candidate_features], dim=-1)

        a = F.relu(self.actor_in(x))
        a = self.actor_res(a)
        logits = self.actor_out(a).squeeze(-1)

        logits = logits.masked_fill(action_mask == 0, -1e9)
        return logits

    def forward_critic(self, global_state):
        # critic根据global_state，计算value和cost
        # value
        v = F.relu(self.value_in(global_state))
        v = self.value_res(v)
        value = self.value_out(v)

        # cost
        c = F.relu(self.cost_in(global_state))
        c = self.cost_res(c)
        cost = self.cost_out(c)

        return value, cost
