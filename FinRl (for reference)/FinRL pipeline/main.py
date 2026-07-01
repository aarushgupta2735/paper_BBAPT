import pandas as pd
from finrl.meta.preprocessor.preprocessors import data_split

from config.config import AppConfig, DirectoryConfig
from src.ML import predict_ml_models, train_ml_models
from src.analysis import (
    build_meta_score,
    build_positive_ratios,
    calculate_meta_score_coefficients,
    calculate_multi_performance_scores,
    calculate_performance_scores,
    calculate_saliency_maps,
)
from src.data_prep import data_prep, setup_directories
from src.drl import (
    build_env_kwargs,
    create_portfolio_env,
    get_baseline_results,
    predict_drl_models,
    train_drl_models,
)
from src.experiment_logging import (
    finish_wandb_run,
    init_wandb_run,
    log_evaluation_results,
    prompt_run_name,
)
from src.plots import (
    plot_cumulative_returns,
    plot_score_histograms,
    plot_sharpe_vs_correlation,
)


APP_CONFIG = AppConfig()
DIRECTORY_CONFIG = DirectoryConfig()


def run_pipeline(
    app_config=APP_CONFIG,
    directory_config=DIRECTORY_CONFIG,
    wandb_run_name=None,
):
    wandb_run = init_wandb_run(wandb_run_name, app_config)

    try:
        setup_directories(directory_config)

        df = data_prep(app_config)
        train = data_split(
            df,
            app_config.data_prep.train_start_date,
            app_config.data_prep.train_end_date,
        )
        trade = data_split(
            df,
            app_config.evaluation.trade_start_date,
            app_config.evaluation.trade_end_date,
        )

        unique_trade_date = trade.date.unique()

        env_kwargs = build_env_kwargs(train, app_config)
        drl_models = train_drl_models(train, env_kwargs, app_config)

        trade_env = create_portfolio_env(trade, env_kwargs)
        baseline_df, baseline_stats, baseline_cumprod = get_baseline_results(app_config)
        drl_results = predict_drl_models(drl_models, trade_env)

        ml_models = train_ml_models(train, app_config)
        ml_predictions = predict_ml_models(df, unique_trade_date, ml_models, app_config)

        meta_q = calculate_saliency_maps(
            trade,
            unique_trade_date,
            drl_models,
            drl_results,
            app_config,
        )
        meta_score_coef = calculate_meta_score_coefficients(
            df,
            unique_trade_date,
            ml_predictions,
            drl_results,
            app_config,
        )
        performance_score = calculate_performance_scores(
            unique_trade_date,
            meta_score_coef,
            meta_q,
            app_config,
        )
        multi_performance_score = calculate_multi_performance_scores(
            unique_trade_date,
            meta_score_coef,
            meta_q,
            app_config,
        )
        meta_score = build_meta_score(ml_predictions, drl_results, baseline_stats)
        positive_ratio, positive_ratio_multi = build_positive_ratios(
            performance_score,
            multi_performance_score,
            meta_score,
        )

        results = {
            "df": df,
            "train": train,
            "trade": trade,
            "baseline": baseline_df,
            "baseline_cumprod": baseline_cumprod,
            "drl_models": drl_models,
            "drl_results": drl_results,
            "ml_models": ml_models,
            "ml_predictions": ml_predictions,
            "meta_q": meta_q,
            "meta_score_coef": meta_score_coef,
            "performance_score": performance_score,
            "multi_performance_score": multi_performance_score,
            "meta_score": meta_score,
            "positive_ratio": positive_ratio,
            "positive_ratio_multi": positive_ratio_multi,
        }
        log_evaluation_results(wandb_run, results)

        time_ind = pd.Series(drl_results["A2C"]["daily_return"].date)
        plot_cumulative_returns(time_ind, drl_results, baseline_cumprod, ml_predictions)
        plot_sharpe_vs_correlation(positive_ratio, positive_ratio_multi)
        plot_score_histograms(performance_score, multi_performance_score)

        return results
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    run_name = prompt_run_name() if APP_CONFIG.logging.log_to_wandb else None
    run_pipeline(wandb_run_name=run_name)
