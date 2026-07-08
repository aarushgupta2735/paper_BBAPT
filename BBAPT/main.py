import os
from dataclasses import asdict
from time import perf_counter

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import TimesNet
from stable_baselines3 import A2C
from gymnasium.utils.env_checker import check_env

from BBAPT.config.config import appConfig
from BBAPT.src.behavioral_map import apply_behavioural_mapping
from BBAPT.src.data_prep import data_for_timesnet, data_prep
from BBAPT.src.portfolio_env import PortfolioEnv
from BBAPT.src.gridsearch import run_grid_search

def init_wandb_run(config):
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging.")
        return None

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "BBAPT"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=os.getenv("WANDB_RUN_NAME") or None,
        mode=os.getenv("WANDB_MODE", "offline"),
        config={
            **asdict(config),
            "n_stocks": config.n_stocks,
            "n_indicators": config.n_indicators,
        },
    )
    return wandb


def log_wandb_metrics(wandb_module, metrics):
    if wandb_module is not None:
        wandb_module.log(metrics)


def main():
    config = appConfig(
        initial_balance=100000,
        allow_short_selling=False,
        technical_indicator_list=["MACD", "RSI_14", "CCI_20", "SMA_20", "EMA_20"],
        transaction_cost=0.001,
        ticker_list=[
            "BTC-USD",
            "ETH-USD",
            "LTC-USD",
            "LINK-USD",
            "BCH-USD",
            "UNI-USD",
            "XLM-USD",
            "FIL-USD",
            "BNB-USD",
            "SOL-USD",
            "XRP-USD",
            "ADA-USD",
            "SHIB-USD",
            "TON-USD",
            "DOGE-USD",
            "AVAX-USD",
            "TRX-USD",
            "DOT-USD",
            "MATIC-USD",
            "ETC-USD",
        ],
        train_starting_date="2020-09-22",
        train_ending_date="2022-06-07",
        test_starting_date="2022-06-08",
        test_ending_date="2023-01-03",
        THRESHOLD_PARAMETER=0.015,
        #oc_upper_threshold=0.01,
        #oc_lower_threshold=-0.01,
        #oc_k_loss=0.1,
        #oc_k_gain=0.1,
        #oc_n=0.725,
        #ra_upper_threshold=0.01,
        #ra_lower_threshold=-0.01,
        #ra_k_loss=0.1,
        #ra_k_gain=0.1,
        #ra_n=1.22,
    )

    oc_params = {
        'oc_upper_threshold': [0.01, 0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12],
        'oc_lower_threshold': [-0.01, -0.02, -0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.09,-0.10,-0.12],
        'oc_k_loss': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'oc_k_gain': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'oc_n': [1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.40],
    }

    ra_params = {
        'ra_upper_threshold': [0.0,0.01, 0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],
        'ra_lower_threshold': [0.0,-0.01, -0.02, -0.03,-0.04,-0.05,-0.06,-0.07,-0.08,-0.09,-0.1],
        'ra_k_loss': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'ra_k_gain': [0.05,0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'ra_n': [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    }
    #run once, find and then directly plug in values
    run_grid_search(config, model_rl, forecast_fn, "ra", ra_params, val_fraction=0.20)
    run_grid_search(config, model_rl, forecast_fn, "oc", oc_params, val_fraction=0.20)

    wandb_module = init_wandb_run(config)
    run_started_at = perf_counter()

    try:
        train_df = data_prep(
            config.train_starting_date,
            config.train_ending_date,
            config.ticker_list,
        )
        test_df = data_prep(
            config.test_starting_date,
            config.test_ending_date,
            config.ticker_list,
        )
        log_wandb_metrics(
            wandb_module,
            {
                "data/train_rows": len(train_df),
                "data/test_rows": len(test_df),
                "data/tickers": len(config.ticker_list),
                "data/features": len(config.technical_indicator_list),
            },
        )

        model = TimesNet(
            h=1,
            input_size=24,
            hidden_size=16,
            conv_hidden_size=32,
            loss=DistributionLoss(distribution="Normal", level=[80, 90]),
            scaler_type="standard",
            learning_rate=1e-3,
            max_steps=100,
            val_check_steps=50,
            early_stop_patience_steps=2,
            enable_checkpointing=True
        )
        nf = NeuralForecast(models=[model], freq="D",)
        nf.fit(df=data_for_timesnet(train_df), val_size=1)
        log_wandb_metrics(
            wandb_module,
            {
                "forecast/train_rows": len(train_df),
                "forecast/fit_completed": 1,
            },
        )

        env = PortfolioEnv(data=train_df, config=config)
        try:
            check_env(env.unwrapped)
            print("Environment passes all checks!")
            log_wandb_metrics(wandb_module, {"env/check_passed": 1})
        except Exception as exc:
            print(f"Environment has issues: {exc}")
            log_wandb_metrics(
                wandb_module,
                {"env/check_passed": 0, "env/check_error": str(exc)},
            )

        print("--- Training RL Model ---")
        train_started_at = perf_counter()
        model_rl = A2C("MlpPolicy", env, verbose=1)
        model_rl.learn(total_timesteps=10000)
        train_duration_seconds = perf_counter() - train_started_at
        print("Training completed!")
        log_wandb_metrics(
            wandb_module,
            {
                "train/total_timesteps": 10000,
                "train/duration_seconds": train_duration_seconds,
            },
        )

        print("--- Running Inference ---")
        test_env = PortfolioEnv(data=test_df, config=config)

        obs, info = test_env.reset()
        done = False
        date = info["date"]

        while not done:
            action, _states = model_rl.predict(obs)
            forecast = nf.predict(df=data_for_timesnet(test_df[test_df["ds"] <= date]))
            avg_return = forecast["TimesNet"].mean()
            action = apply_behavioural_mapping(action, avg_return, test_df, date, config)
            obs, reward, done, truncated, info = test_env.step(action)
            date = info["date"]

        final_balance = float(info["balance"])
        sharpe_ratio = float(info['sharpe_ratio'])
        rolling_sharpe = float(info['rolling_sharpe_ratio'])
        FCR = float(info['FCR'])
        ARR = float(info['ARR'])
        AV = float(info['AV'])
        MDD = float(info['MDD'])
        
        print(f"Inference completed! Final portfolio balance: {final_balance:.2f}")
        log_wandb_metrics(
            wandb_module,
            {
                "eval/final_balance": final_balance,
                "eval/final_profit": final_balance - config.initial_balance,
                "run/duration_seconds": perf_counter() - run_started_at,
                "eval/rolling_sharpe_ratio": rolling_sharpe,
                "eval/sharpe_ratio": sharpe_ratio,
                "eval/FCR" :FCR,
                "eval/ARR" :ARR,
                "eval/AV" :AV,
                "eval/MDD" :MDD,
                "eval/weights" :info['weights']
            },
        )
    except Exception as exc:
        log_wandb_metrics(wandb_module, {"run/error": str(exc)})
        raise
    finally:
        if wandb_module is not None:
            wandb_module.finish()


if __name__ == "__main__":
    main()
