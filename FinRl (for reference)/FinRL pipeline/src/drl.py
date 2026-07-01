import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import (
    backtest_stats,
    convert_daily_return_to_pyfolio_ts,
    get_baseline,
    get_daily_return,
)
from pyfolio import timeseries

from src.stock_portfolio_env import StockPortfolioEnv


def build_env_kwargs(train_data, app_config):
    stock_dimension = len(train_data.tic.unique())
    tech_indicator_list = app_config.environment.tech_indicator_list
    print(f"Stock Dimension: {stock_dimension}, State Space: {stock_dimension}")
    print(f"Feature Dimension: {len(tech_indicator_list)}")

    return {
        "hmax": app_config.environment.hmax,
        "initial_amount": app_config.environment.initial_amount,
        "transaction_cost_pct": app_config.environment.transaction_cost_pct,
        "state_space": stock_dimension,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": app_config.environment.reward_scaling,
    }


def create_portfolio_env(data, env_kwargs):
    return StockPortfolioEnv(df=data, **env_kwargs)


def train_drl_models(train_data, env_kwargs, app_config):
    e_train_gym = create_portfolio_env(train_data, env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model(
        model_name="a2c",
        model_kwargs=app_config.training.a2c_params,
    )
    trained_a2c = agent.train_model(
        model=model_a2c,
        tb_log_name="a2c",
        total_timesteps=app_config.training.a2c_total_timesteps,
    )

    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model(
        "ppo",
        model_kwargs=app_config.training.ppo_params,
    )
    trained_ppo = agent.train_model(
        model=model_ppo,
        tb_log_name="ppo",
        total_timesteps=app_config.training.ppo_total_timesteps,
    )

    return {"A2C": trained_a2c, "PPO": trained_ppo}


def get_baseline_results(app_config):
    evaluation = app_config.evaluation
    baseline_df = get_baseline(
        ticker=evaluation.baseline_ticker,
        start=evaluation.baseline_start_date,
        end=evaluation.baseline_end_date,
    )
    baseline_stats = backtest_stats(baseline_df, value_col_name="close")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    baseline_cumprod = (baseline_returns + 1).cumprod() - 1
    return baseline_df, baseline_stats, baseline_cumprod


def extract_weights(drl_actions_list):
    weight_rows = {"date": [], "weights": []}
    for i in range(len(drl_actions_list)):
        date = drl_actions_list.index[i]
        tic_list = list(drl_actions_list.columns)
        weights_list = (
            drl_actions_list.reset_index()[list(drl_actions_list.columns)]
            .iloc[i]
            .values
        )
        weight_rows["date"].append(date)
        weight_rows["weights"].append(
            pd.DataFrame({"tic": tic_list, "weight": weights_list})
        )
    return pd.DataFrame(weight_rows)


def predict_drl_models(models, trade_env):
    daily_return_a2c, actions_a2c = DRLAgent.DRL_prediction(
        model=models["A2C"],
        environment=trade_env,
    )
    daily_return_ppo, actions_ppo = DRLAgent.DRL_prediction(
        model=models["PPO"],
        environment=trade_env,
    )

    perf_func = timeseries.perf_stats
    strat_a2c = convert_daily_return_to_pyfolio_ts(daily_return_a2c)
    strat_ppo = convert_daily_return_to_pyfolio_ts(daily_return_ppo)

    return {
        "A2C": {
            "daily_return": daily_return_a2c,
            "actions": actions_a2c,
            "cumprod": (daily_return_a2c.daily_return + 1).cumprod() - 1,
            "stats": perf_func(
                returns=strat_a2c,
                factor_returns=strat_a2c,
                positions=None,
                transactions=None,
                turnover_denom="AGB",
            ),
            "weights": extract_weights(actions_a2c),
        },
        "PPO": {
            "daily_return": daily_return_ppo,
            "actions": actions_ppo,
            "cumprod": (daily_return_ppo.daily_return + 1).cumprod() - 1,
            "stats": perf_func(
                returns=strat_ppo,
                factor_returns=strat_ppo,
                positions=None,
                transactions=None,
                turnover_denom="AGB",
            ),
            "weights": extract_weights(actions_ppo),
        },
    }
