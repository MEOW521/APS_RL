import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from aps_rl.utils.config import APSConfig

class SchedulingEnv(gym.Env):
    def __init__(self, data_path: str, config_path: str):
        super().__init__()
        # 配置项
        self.config = APSConfig(config_path)
        self.max_continuous_model_qty = float(self.config.get("env.max_continuous_model_qty", 240.0))
        self.max_continuous_biw_qty = float(self.config.get("env.max_continuous_biw_qty", 60.0))
        self.max_continuous_mat_qty = float(self.config.get("env.max_continuous_mat_qty", 30.0))
        self.max_model_biw_types = float(self.config.get("env.max_model_biw_types", 3.0))
        self.max_biw_mat_types = float(self.config.get("env.max_biw_mat_types", 2.0))


        # 读取数据: 车型，白车身，整车物料号，计划上线日期，计划产量
        self.df = pd.read_csv(data_path, encoding='utf-8')

        # 整车物料号为最小生产单位，以此聚合
        self.df_agg = self.df.groupby('整车物料号').agg(
            {
                '车型': 'first',
                '白车身': 'first',
                '计划上线日期': 'first',
                '计划产量': 'sum'
            }
        ).reset_index()

        # 动作空间：整车物料号
        # agent选择一个整车物料号，环境选择生产的数量
        self.num_actions = len(self.df_agg)
        self.action_space = spaces.Discrete(self.num_actions)

        # 状态空间：全局状态、候选特征、动作掩码
        self.observation_space = spaces.Dict({
            "global_state": spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32),
            "candidate_features": spaces.Box(0.0, 1.0, shape=(self.num_actions, 9), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(self.num_actions,), dtype=np.float32)
        })

        # 创建实例时初始化环境
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 维护每个物料号剩余订单数量
        self.remaining_qty = self.df_agg['计划产量'].values.copy()

        # 动作掩码：1表示可以生产，0表示不能生产
        self.action_mask = np.ones(self.num_actions, dtype=np.float32)
        # 记录上一个生产的车型、白车身、整车物料号
        self.last_model, self.last_biw, self.last_mat = None, None, None
        # 车型、白车身、物料号连续数量
        self.model_continuous_qty = 0
        self.biw_continuous_qty = 0
        self.mat_continuous_qty = 0
        # 
        self.model_biw_types = set()
        self.biw_mat_types = set()
        
        self.step_cnt = 0

        return self._get_obs(), {}


    def _get_obs(self):
        """获取当前环境状态信息
        返回值:
        {
            "global_state": 全局状态,
            "candidate_features": 候选特征,
            "action_mask": 动作掩码,
        }
        """
        # 全局状态
        global_state = np.zeros(5, dtype=np.float32)
        global_state[0] = min(self.model_continuous_qty / self.max_continuous_model_qty, 1.0) # 当前车型连续数量/max_continuous_model_qty
        global_state[1] = min(self.biw_continuous_qty / self.max_continuous_biw_qty, 1.0) # 当前白车身连续数量/max_continuous_biw_qty
        global_state[2] = min(self.mat_continuous_qty / self.max_continuous_mat_qty, 1.0) # 当前物料号连续数量/max_continuous_mat_qty
        global_state[3] = min(len(self.model_biw_types) / self.max_model_biw_types, 1.0) # 车型下白车身切换数量/max_model_biw_types
        global_state[4] = min(len(self.biw_mat_types) / self.max_biw_mat_types, 1.0) # 白车身下物料号切换数量/max_biw_mat_types

        # 候选特征
        candidate_features = np.zeros((self.num_actions, 9), dtype=np.float32)
        for idx in range(self.num_actions):
            if self.action_mask[idx] == 0: continue

            order = self.df_agg.iloc[idx]
            remain = self.remaining_qty[idx]

            # 是否连续生产同一车型、白车身、物料号
            is_same_model = 1.0 if self.last_model == order['车型'] else 0.0
            is_same_biw = 1.0 if self.last_biw == order['白车身'] else 0.0
            is_same_mat = 1.0 if self.last_mat == order['整车物料号'] else 0.0

            # 计算最大订单释放量
            model_limit = self.max_continuous_model_qty - self.model_continuous_qty if is_same_model else self.max_continuous_model_qty
            biw_limit = self.max_continuous_biw_qty - self.biw_continuous_qty if is_same_biw else self.max_continuous_biw_qty
            mat_limit = self.max_continuous_mat_qty - self.mat_continuous_qty if is_same_mat else self.max_continuous_mat_qty
            max_release_qty = min(model_limit, biw_limit, mat_limit)

            if max_release_qty > 0:
                pred_release_qty = min(remain, max_release_qty)
            else:
                pred_release_qty = min(remain, 30.0)

            
            # 判断是否会违规
            vio_model_qty = 1.0 if (is_same_model and self.model_continuous_qty + pred_release_qty > self.max_continuous_model_qty) else 0.0
            vio_biw_qty = 1.0 if (is_same_biw and self.biw_continuous_qty + pred_release_qty > self.max_continuous_biw_qty) else 0.0
            vio_mat_qty = 1.0 if (is_same_mat and self.mat_continuous_qty + pred_release_qty > self.max_continuous_mat_qty) else 0.0
            vio_model_biw = 1.0 if (is_same_model and not is_same_biw and order['白车身'] not in self.model_biw_types and len(self.model_biw_types) >= self.max_model_biw_types) else 0.0
            vio_biw_mat = 1.0 if (is_same_biw and not is_same_mat and order['整车物料号'] not in self.biw_mat_types and len(self.biw_mat_types) >= self.max_biw_mat_types) else 0.0

            candidate_features[idx] = [
                is_same_model, is_same_biw, is_same_mat, \
                pred_release_qty/min(self.max_continuous_model_qty, self.max_continuous_biw_qty, self.max_continuous_mat_qty), \
                vio_model_qty, vio_biw_qty, vio_mat_qty, vio_model_biw, vio_biw_mat
            ]
        
        return {
            "global_state": global_state,
            "candidate_features": candidate_features,
            "action_mask": np.copy(self.action_mask)
        }
    
    def step(self, action: int):
        order = self.df_agg.iloc[action]
        remain = self.remaining_qty[action]

        is_same_model = 1.0 if self.last_model == order['车型'] else 0.0
        is_same_biw = 1.0 if self.last_biw == order['白车身'] else 0.0
        is_same_mat = 1.0 if self.last_mat == order['整车物料号'] else 0.0

        model_limit = self.max_continuous_model_qty - self.model_continuous_qty if is_same_model else self.max_continuous_model_qty
        biw_limit = self.max_continuous_biw_qty - self.biw_continuous_qty if is_same_biw else self.max_continuous_biw_qty
        mat_limit = self.max_continuous_mat_qty - self.mat_continuous_qty if is_same_mat else self.max_continuous_mat_qty
        max_release_qty = min(model_limit, biw_limit, mat_limit)

        if max_release_qty > 0:
            release_qty = min(remain, max_release_qty)
        else:
            release_qty = min(remain, 30.0)
        
        self.remaining_qty[action] -= release_qty

        # reward
        reward = 0.0
        if self.step_cnt > 0:
            if is_same_mat: reward += 2.0
            elif is_same_biw: reward += 1.0
            elif is_same_model: reward += 0.5
        else:
            reward += 1.0

        # 状态更新
        if is_same_model:
            self.model_continuous_qty += release_qty
            self.model_biw_types.add(order['白车身'])
        else:
            self.model_continuous_qty = release_qty
            self.model_biw_types = {order['白车身']}
        if is_same_biw:
            self.biw_continuous_qty += release_qty
            self.biw_mat_types.add(order['整车物料号'])
        else:
            self.biw_continuous_qty = release_qty
            self.biw_mat_types = {order['整车物料号']}
        if is_same_mat:
            self.mat_continuous_qty += release_qty
        else:
            self.mat_continuous_qty = release_qty

        # cost
        costs = np.zeros(5, dtype=np.float32)
        if self.model_continuous_qty > self.max_continuous_model_qty:
            costs[0] = 1.0
        if self.biw_continuous_qty > self.max_continuous_biw_qty:
            costs[1] = 1.0
        if self.mat_continuous_qty > self.max_continuous_mat_qty:
            costs[2] = 1.0
        if len(self.model_biw_types) > self.max_model_biw_types:
            costs[3] = 1.0
        if len(self.biw_mat_types) > self.max_biw_mat_types:
            costs[4] = 1.0

        # 动作掩码，如果该物料号已经排完，需要掩掉
        if self.remaining_qty[action] == 0:
            self.action_mask[action] = 0.0
        
        # 记录本次释放信息
        self.last_model = order['车型']
        self.last_biw = order['白车身']
        self.last_mat = order['整车物料号']
        self.step_cnt += 1

        terminated = bool(np.sum(self.remaining_qty) <= 1e-6)
        truncated = False
        info = {"release_qty": float(release_qty), "cost": costs, "cost_sum": float(np.sum(costs))}
        
        return self._get_obs(), float(reward), terminated, truncated, info


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    env = SchedulingEnv(data_path="../data/train_data.csv", config_path="../config/config.json")
    check_env(env, skip_render_check=True)
