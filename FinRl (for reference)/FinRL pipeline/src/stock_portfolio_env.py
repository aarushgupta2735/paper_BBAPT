import os
import copy
from datetime import datetime as dt
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv


class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment for Gymnasium."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.state_space + len(self.tech_indicator_list),
                self.state_space,
            ),
        )

        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.reward = 0  # FIX 5: initialize reward

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        self.reward = 0  # FIX 5: reset reward each step

        if self.terminal:
            df_ret = pd.DataFrame(self.portfolio_return_memory)
            df_ret.columns = ["daily_return"]
            plt.plot(df_ret.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")

            # FIX 3: return (obs, reward, terminated, truncated, info)
            return self.state, self.reward, self.terminal, False, {}

        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )
            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = new_portfolio_value

        # FIX 3: return (obs, reward, terminated, truncated, info)
        return self.state, self.reward, self.terminal, False, {}

    def reset(self, seed=None, options=None):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0,
        )
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.reward = 0
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim] ##starts from equal allocation
        self.date_memory = [self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        return numerator / denominator

    def save_asset_memory(self):
        return pd.DataFrame({
            "date": self.date_memory,
            "daily_return": self.portfolio_return_memory,
        })

    def save_action_memory(self):
        df_date = pd.DataFrame(self.date_memory, columns=["date"])
        df_actions = pd.DataFrame(self.actions_memory)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        # FIX 6: gymnasium removed gym.utils.seeding; use numpy directly
        rng = np.random.default_rng(seed)
        self.np_random = rng
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

