import os
import numpy as np
from dataclasses import asdict

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction, EqualWeighted, RiskBudgeting
from skfolio.measures import RiskMeasure
from BBAPT.config.config import appConfig
from BBAPT.src.data_prep import data_prep

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

def maximum_sharpe_ratio_model(train_df, test_df):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
    )
    model.fit(train_df)
    action = model.predict(test_df)
    
    action.compounded = True 
    return {
        "weights": action.weights,
        "portfolio_sharpe_ratio": action.sharpe_ratio,
        "ARR": action.annualized_mean,
        "MDD": action.max_drawdown,
        "AV": action.annualized_variance,
        "FCR": action.cumulative_returns[-1]
    }

def minimum_tail_risk_model(train_df, test_df):
    model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)
    model.fit(train_df)
    action = model.predict(test_df)
    
    action.compounded = True
    return {
        "weights": action.weights,
        "portfolio_sharpe_ratio": action.sharpe_ratio,
        "ARR": action.annualized_mean,
        "MDD": action.max_drawdown,
        "AV": action.annualized_variance,
        "FCR": action.cumulative_returns[-1]
    }

def minimum_variance_portfolio(train_df, test_df):
    model = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)
    model.fit(train_df)
    action = model.predict(test_df)
    
    action.compounded = True
    return {
        "weights": action.weights,
        "portfolio_sharpe_ratio": action.sharpe_ratio,
        "ARR": action.annualized_mean,
        "MDD": action.max_drawdown,
        "AV": action.annualized_variance,
        "FCR": action.cumulative_returns[-1]
    }

def naive_equal_weighted_portfolio(train_df, test_df):
    # Using skfolio's built-in EqualWeighted estimator
    model = EqualWeighted()
    model.fit(train_df)
    action = model.predict(test_df)
    
    action.compounded = True
    return {
        "weights": action.weights,
        "portfolio_sharpe_ratio": action.sharpe_ratio,
        "ARR": action.annualized_mean,
        "MDD": action.max_drawdown,
        "AV": action.annualized_variance,
        "FCR": action.cumulative_returns[-1]
    }

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
    
    wandb_module = init_wandb_run(config)
    
    tickers_attr = getattr(config, "tickers", getattr(config, "ticker_list", None))
    train_df = data_prep(
        config.train_starting_date,
        config.train_ending_date,
        tickers_attr,
    )
    test_df = data_prep(
        config.test_starting_date,
        config.test_ending_date,
        tickers_attr,
    )
    
    # skfolio expects wide-format data (dates as index, tickers as columns, returns as values)
    train_returns = train_df.pivot(index='ds', columns='unique_id', values='y')[tickers_attr].dropna()
    test_returns = test_df.pivot(index='ds', columns='unique_id', values='y')[tickers_attr].dropna()

    maximum_sharpe_ratio_model_result = maximum_sharpe_ratio_model(train_returns, test_returns)
    minimum_tail_risk_model_result = minimum_tail_risk_model(train_returns, test_returns)
    equally_weighted_portfolio_result = naive_equal_weighted_portfolio(train_returns, test_returns)
    minimum_variance_portfolio_result = minimum_variance_portfolio(train_returns, test_returns)

    print("\n-----Running Baseline Models ----\n")

    ## Add metrics FCR, ARR, AV, MDD
    models = {
        "Maximum Sharpe Ratio": maximum_sharpe_ratio_model_result,
        "Minimum Tail Risk": minimum_tail_risk_model_result,
        "Equally Weighted Portfolio": equally_weighted_portfolio_result,
        "Minimum Variance Portfolio": minimum_variance_portfolio_result
    }
    
    metrics_to_log = {}

    for name, result in models.items():
        print(f"---{name}---")
        print(f"Portfolio_sharpe_ratio: {result.get('portfolio_sharpe_ratio', 'N/A')}")
        print(f"Weights: {result.get('weights', 'N/A')}")
        print(f"FCR: {result.get('FCR', 'N/A')}")
        print(f"ARR: {result.get('ARR', 'N/A')}")
        print(f"AV: {result.get('AV', 'N/A')}")
        print(f"MDD: {result.get('MDD', 'N/A')}\n")
        
        for k, v in result.items():
            if v is not None:
                if isinstance(v, np.ndarray):
                    metrics_to_log[f"{name}/{k}"] = v.tolist()
                else:
                    metrics_to_log[f"{name}/{k}"] = v

    log_wandb_metrics(wandb_module, metrics_to_log)
    
    if wandb_module is not None and wandb_module.run is not None:
        wandb_module.run.summary.update(metrics_to_log)
    
    if wandb_module is not None:
        wandb_module.finish()

if __name__ == "__main__":
    main()
    
    
    