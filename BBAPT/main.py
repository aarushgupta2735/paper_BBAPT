import os
from dataclasses import asdict
from time import perf_counter

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import TimesNet
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.utils.env_checker import check_env

from BBAPT.config.config import appConfig
from BBAPT.src.behavioral_map import apply_behavioural_mapping
from BBAPT.src.data_prep import data_for_timesnet, data_prep
from BBAPT.src.portfolio_env import PortfolioEnv

from BBAPT.src.a2c import PortfolioActorCriticPolicy
import torch.nn as nn



def init_wandb_run(config):
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging.")
        return None

    run_name = os.getenv("WANDB_RUN_NAME")
    if not run_name:
        run_name = input(f"Enter wandb run name (leave empty for default: {config.default_run_name}): ").strip()
        if not run_name:
            run_name = config.default_run_name

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "BBAPT-baseline"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=run_name,
        mode=os.getenv("WANDB_MODE", "offline"),
        sync_tensorboard=True,
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
        tickers=["BTC-USD", "ETH-USD", "LTC-USD", "LINK-USD", "BCH-USD", "UNI-USD", "XLM-USD", "FIL-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "SHIB-USD", "TON-USD", "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "ETC-USD"],
        train_starting_date="2020-09-22",
        train_ending_date="2022-06-07",
        test_starting_date="2022-06-08",
        test_ending_date="2023-01-03",

        #config based on grid_search 
        THRESHOLD_PARAMETER=0.015,
        oc_upper_threshold=0.01,
        oc_lower_threshold=-0.01,
        oc_k_loss=0.1,
        oc_k_gain=0.1,
        oc_n=0.725,
        ra_upper_threshold=0.01,
        ra_lower_threshold=-0.01,
        ra_k_loss=0.1,
        ra_k_gain=0.1,
        ra_n=1.22,
        tn_early_stop_patience_steps=-1,
        rl_algorithm="A2C",
    )

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
            h=config.tn_h,
            input_size=config.tn_input_size,
            hidden_size=config.tn_hidden_size,
            conv_hidden_size=config.tn_conv_hidden_size,
            loss=DistributionLoss(distribution=config.tn_loss_distribution, level=config.tn_loss_level),
            scaler_type=config.tn_scaler_type,
            learning_rate=config.tn_learning_rate,
            max_steps=config.tn_max_steps,
            val_check_steps=config.tn_val_check_steps,
            early_stop_patience_steps=config.tn_early_stop_patience_steps,
            enable_checkpointing=config.tn_enable_checkpointing
        )
        nf = NeuralForecast(models=[model], freq="D",)
        #nf.fit(df=data_for_timesnet(train_df), val_size=1)

        log_wandb_metrics(
            wandb_module,
            {
                "forecast/train_rows": len(train_df),
                "forecast/fit_completed": 1,
            },
        )

        env = PortfolioEnv(data=train_df, config=config, already_weight = False)
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

        algo_class = {"A2C": A2C, "PPO": PPO, "DDPG": DDPG}[config.rl_algorithm]
        model_hash = config.get_rl_config_hash()
        model_path = f"./checkpoints/main/rl_model_{config.rl_algorithm}_{model_hash}_final.zip"
        
        if os.path.exists(model_path):
            print(f"--- Loading existing {config.rl_algorithm} Model from {model_path} ---")
            model_rl = algo_class.load(model_path, env=env)
        else:
            print(f"--- Training {config.rl_algorithm} Model ---")
        
            train_started_at = perf_counter()
            
            if config.rl_algorithm == "A2C":
                model_rl = A2C(PortfolioActorCriticPolicy, env, verbose=1,
                    tensorboard_log="./wandb_tb_logs/",
                    policy_kwargs=dict(
                    net_arch=dict(pi=[320, 160, 80], vf=[64, 16, 4]),
                    activation_fn=nn.Tanh,
                ),)
            elif config.rl_algorithm == "PPO":
                model_rl = PPO(PortfolioActorCriticPolicy, env, verbose=1,
                    tensorboard_log="./wandb_tb_logs/",
                    policy_kwargs=dict(
                    net_arch=dict(pi=[320, 160, 80], vf=[64, 16, 4]),
                    activation_fn=nn.Tanh,
                ),)
            elif config.rl_algorithm == "DDPG":
                model_rl = DDPG("MlpPolicy", env, verbose=1,
                    tensorboard_log="./wandb_tb_logs/",
                    policy_kwargs=dict(
                    net_arch=dict(pi=[320, 160, 80], qf=[64, 16, 4]),
                    activation_fn=nn.Tanh,
                ),)

            checkpoint_callback = CheckpointCallback(
                save_freq=2500,
                save_path="./checkpoints/main/",
                name_prefix=f"rl_model_{config.rl_algorithm}_{model_hash}"
            )
            model_rl.learn(total_timesteps=10000, callback=checkpoint_callback)
            model_rl.save(model_path.replace(".zip", ""))
            train_duration_seconds = perf_counter() - train_started_at
            print("Training completed!")
            log_wandb_metrics(
                wandb_module,
                {
                    "train/total_timesteps": 10000,
                    "train/duration_seconds": train_duration_seconds,
                },
            )
        print("--- Using Configured Behavioral Hyperparameters ---")
        print(f"oc_upper_threshold = {config.oc_upper_threshold}")
        print(f"oc_lower_threshold = {config.oc_lower_threshold}")
        print(f"oc_k_loss = {config.oc_k_loss}")
        print(f"oc_k_gain = {config.oc_k_gain}")
        print(f"oc_n = {config.oc_n}")
        print(f"ra_upper_threshold = {config.ra_upper_threshold}")
        print(f"ra_lower_threshold = {config.ra_lower_threshold}")
        print(f"ra_k_loss = {config.ra_k_loss}")
        print(f"ra_k_gain = {config.ra_k_gain}")
        print(f"ra_n = {config.ra_n}")

        print("--- Running Inference ---")
        full_df = pd.concat([train_df, test_df])
        test_size = test_df['ds'].nunique()  # number of test time steps
        
        cv_df = nf.cross_validation(df=data_for_timesnet(full_df), test_size=test_size, step_size=config.tn_h, n_windows=None)
        
        test_env = PortfolioEnv(data=test_df, config=config, already_weight = True)

        obs, info = test_env.reset()
        done = False
        date = info["date"]

        while not done:
            action, _states = model_rl.predict(obs)
            
            # Using the precomputed cross-validation predictions
            forecast = cv_df[cv_df["cutoff"] == date]
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
