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
import torch.nn as nn

from BBAPT.config.config import appConfig
from BBAPT.src.behavioral_map import apply_behavioural_mapping
from BBAPT.src.data_prep import data_for_timesnet, data_prep
from BBAPT.src.portfolio_env import PortfolioEnv
from BBAPT.src.gridsearch import run_grid_search
from BBAPT.src.a2c import PortfolioActorCriticPolicy

def init_wandb_run(config):
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging.")
        return None

    run_name = os.getenv("WANDB_RUN_NAME")
    if not run_name:
        run_name = input(f"Enter wandb run name (leave empty for default: gridsearch_{config.default_run_name}): ").strip()
        if not run_name:
            run_name = f"gridsearch_{config.default_run_name}"

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "BBAPT-gridsearch"),
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
        THRESHOLD_PARAMETER=0.015,
        tn_early_stop_patience_steps=-1,
        rl_algorithm="A2C",
    )

    wandb_module = init_wandb_run(config)
    run_started_at = perf_counter()

    try:
        full_train_df = data_prep(
            config.train_starting_date,
            config.train_ending_date,
            config.ticker_list,
        )
        
        # Chronological 80-20 split on train data
        dates = sorted(full_train_df['ds'].unique())
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]
        
        train_df = full_train_df[full_train_df['ds'] < split_date].copy()
        val_df = full_train_df[full_train_df['ds'] >= split_date].copy()
        
        log_wandb_metrics(
            wandb_module,
            {
                "data/train_rows": len(train_df),
                "data/val_rows": len(val_df),
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
        model_path = f"./checkpoints/gridsearch/rl_model_{config.rl_algorithm}_final.zip"
        
        if os.path.exists(model_path):
            print(f"--- Loading existing {config.rl_algorithm} Model from {model_path} ---")
            model_rl = algo_class.load(model_path, env=env)
        else:
            print(f"--- Training {config.rl_algorithm} Model for Grid Search ---")
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
                save_path="./checkpoints/gridsearch/",
                name_prefix=f"rl_model_{config.rl_algorithm}"
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

        print("--- Computing Validation Forecasts ---")
        val_test_size = val_df['ds'].nunique()
        val_cv_df = nf.cross_validation(df=data_for_timesnet(full_train_df), test_size=val_test_size, step_size=config.tn_h, n_windows=None)

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

        print("--- Tuning Behavioral Hyperparameters ---")
        print("Tuning Risk Aversion...")
        best_ra_config, ra_results = run_grid_search(config, model_rl, val_df, val_cv_df, "ra", ra_params)
        if wandb_module is not None:
            ra_table = wandb_module.Table(columns=list(ra_params.keys()) + ["val_sharpe"])
            for param_dict, sharpe in ra_results:
                row = [param_dict[k] for k in ra_params.keys()] + [sharpe]
                ra_table.add_data(*row)
            wandb_module.log({"ra_grid_search_results": ra_table})
        
        print("Tuning Overconfidence...")
        best_oc_config, oc_results = run_grid_search(config, model_rl, val_df, val_cv_df, "oc", oc_params)
        if wandb_module is not None:
            oc_table = wandb_module.Table(columns=list(oc_params.keys()) + ["val_sharpe"])
            for param_dict, sharpe in oc_results:
                row = [param_dict[k] for k in oc_params.keys()] + [sharpe]
                oc_table.add_data(*row)
            wandb_module.log({"oc_grid_search_results": oc_table})
        
        if wandb_module is not None:
            wandb_module.config.update({"best_ra_params": ra_results[0][0], "best_oc_params": oc_results[0][0]}, allow_val_change=True)
            wandb_module.summary["best_ra_sharpe"] = ra_results[0][1]
            wandb_module.summary["best_oc_sharpe"] = oc_results[0][1]
        
        print("\n================ GRID SEARCH COMPLETED ================")
        print("Best Risk Aversion Parameters:")
        print(ra_results[0][0])
        print("Best Overconfidence Parameters:")
        print(oc_results[0][0])
        print("========================================================\n")
        
        # Save best hyperparameters to CSV
        import csv
        from datetime import datetime
        
        csv_file = "best_hyperparameters.csv"
        file_exists = os.path.exists(csv_file)
        
        best_ra = ra_results[0][0]
        best_oc = oc_results[0][0]
        
        row_dict = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "RL_Algorithm": config.rl_algorithm,
            "Config_Hash": model_hash,
            "RA_Val_Sharpe": ra_results[0][1],
            "OC_Val_Sharpe": oc_results[0][1]
        }
        row_dict.update(best_ra)
        row_dict.update(best_oc)
        
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
        print(f"Appended optimal parameters to {csv_file}")
        
    finally:
        if wandb_module is not None:
            wandb_module.finish()

if __name__ == "__main__":
    main()
