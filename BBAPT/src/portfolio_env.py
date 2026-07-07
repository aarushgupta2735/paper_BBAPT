##Building the Portfolio using DRL (A2C and PPO)
import gymnasium as gym
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, data, config):
        self.data = data
        self.ticker_list = config.ticker_list
        self.n_assets = len(config.ticker_list)
        self.balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.n_technical_indicators = len(config.technical_indicator_list) if config.technical_indicator_list is not None else 0
        self.technical_indicators_list = config.technical_indicator_list if config.technical_indicator_list is not None else []
        self.allow_short_selling = config.allow_short_selling
        self.transaction_cost = config.transaction_cost
        self.THRESHOLD_PARAMETER = config.THRESHOLD_PARAMETER
        self.oc_upper_threshold = config.oc_upper_threshold
        self.oc_lower_threshold = config.oc_lower_threshold
        self.oc_k_loss = config.oc_k_loss
        self.oc_k_gain = config.oc_k_gain
        self.oc_n = config.oc_n
        self.ra_upper_threshold = config.ra_upper_threshold
        self.ra_lower_threshold = config.ra_lower_threshold
        self.ra_k_loss = config.ra_k_loss
        self.ra_k_gain = config.ra_k_gain
        self.ra_n = config.ra_n

        self.dates = sorted(self.data['ds'].unique().tolist())
        self.initial_date = self.dates[1]
        self.date = self.initial_date
        self.current_step = 0
        self.n_steps = len(self.dates)

        self.action_space = gym.spaces.Box(low=-1 if self.allow_short_selling else 0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.prev_action = np.ones(self.n_assets) / self.n_assets
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets, 1 + self.n_technical_indicators), dtype=np.float32)

    def _get_observation(self, date):
        current_row = self.data[self.data['ds'] == date].sort_values('unique_id')
        features = ['y'] + self.technical_indicators_list
        return current_row[features].to_numpy(dtype=np.float32)

    def step(self, action):
        current_row = self.data[self.data['ds'] == self.date].sort_values('unique_id')
        current_return = current_row['y'].to_numpy(dtype=np.float32)
        std_deviation = current_row['std_dev'].to_numpy(dtype=np.float32)

        profit = (np.multiply(action, current_return).sum()) - (np.abs(action - self.prev_action) * self.transaction_cost).sum()
        reward = profit / std_deviation.mean()
        self.balance += profit

        self.current_step += 1
        done = self.current_step >= self.n_steps

        if not done:
            self.date = self.dates[self.current_step]
            next_observation = self._get_observation(self.date)
        else:
            next_observation = np.zeros((self.n_assets, 1 + self.n_technical_indicators), dtype=np.float32)

        self.prev_action = action
        return next_observation, reward, done, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_action = np.ones(self.n_assets) / self.n_assets
        self.current_step = 0
        self.date = self.initial_date
        self.balance = self.initial_balance
        return self._get_observation(self.date), self._get_info()

    def _get_info(self):
        return {'balance': self.balance, 'date': self.date}