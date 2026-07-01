from dataclasses import asdict

import pandas as pd


def prompt_run_name():
    run_name = input("Enter W&B run name: ").strip()
    return run_name or None


def init_wandb_run(run_name, app_config):
    if not app_config.logging.log_to_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` "
            "or set AppConfig.logging.log_to_wandb=False."
        ) from exc

    if app_config.logging.wandb_mode == "online":
        wandb.login()

    return wandb.init(
        project=app_config.logging.wandb_project,
        entity=app_config.logging.wandb_entity,
        name=run_name,
        mode=app_config.logging.wandb_mode,
        config=asdict(app_config),
    )


def log_evaluation_results(wandb_run, results):
    if wandb_run is None:
        return

    import wandb

    meta_score = results["meta_score"]
    positive_ratio = results["positive_ratio"]
    positive_ratio_multi = results["positive_ratio_multi"]
    performance_score = results["performance_score"]
    multi_performance_score = results["multi_performance_score"]

    wandb.log({
        "tables/meta_score": wandb.Table(dataframe=meta_score),
        "tables/single_step_correlations": wandb.Table(dataframe=performance_score),
        "tables/multi_step_correlations": wandb.Table(dataframe=multi_performance_score),
        "tables/positive_ratio": wandb.Table(dataframe=positive_ratio),
        "tables/positive_ratio_multi": wandb.Table(dataframe=positive_ratio_multi),
    })
    wandb.log(_meta_score_metrics(meta_score))
    wandb.log(_correlation_metrics(positive_ratio, positive_ratio_multi))
    wandb.log(_final_return_metrics(results))


def finish_wandb_run(wandb_run):
    if wandb_run is not None:
        wandb_run.finish()


def _meta_score_metrics(meta_score):
    metric_names = {
        "Annual return": "annual_return",
        "Annual volatility": "annual_volatility",
        "Max drawdown": "max_drawdown",
        "Sharpe ratio": "sharpe_ratio",
        "Calmar ratio": "calmar_ratio",
    }
    metrics = {}

    for _, row in meta_score.iterrows():
        algo = _clean_key(row["Algorithm"])
        for column, metric_name in metric_names.items():
            metrics[f"eval/{algo}/{metric_name}"] = row[column]

    return metrics


def _correlation_metrics(positive_ratio, positive_ratio_multi):
    metrics = {}
    for _, row in positive_ratio.iterrows():
        algo = _clean_key(row["algo"])
        metrics[f"correlation/{algo}/single_step_avg"] = row[
            "avg_correlation_coefficient"
        ]
        metrics[f"correlation/{algo}/sharpe_ratio"] = row["Sharpe Ratio"]

    for _, row in positive_ratio_multi.iterrows():
        algo = _clean_key(row["algo"])
        metrics[f"correlation/{algo}/multi_step_avg"] = row[
            "avg_correlation_coefficient"
        ]

    return metrics


def _final_return_metrics(results):
    metrics = {
        "returns/DJI/final_cumulative_return": _last_value(
            results["baseline_cumprod"]
        ),
    }

    for algo, algo_results in results["drl_results"].items():
        metrics[f"returns/{_clean_key(algo)}/final_cumulative_return"] = _last_value(
            algo_results["cumprod"]
        )

    for algo, prediction in results["ml_predictions"].items():
        if algo == "Reference Model":
            continue
        metrics[f"returns/{_clean_key(algo)}/final_cumulative_return"] = _last_value(
            prediction[2]
        )

    return metrics


def _last_value(values):
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return series.iloc[-1]


def _clean_key(value):
    return str(value).replace(" ", "_").replace("/", "_")
