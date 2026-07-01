import copy

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch


def calculate_gradient(
    model,
    interpolated_input,
    actions,
    feature_idx,
    stock_idx,
    stock_dimension,
    feature_dimension,
    h=1e-1,
):
    forward_input = interpolated_input.copy()
    forward_input[feature_idx + stock_dimension][stock_idx] += h
    forward_q = model.policy.evaluate_actions(
        torch.FloatTensor(forward_input).reshape(
            -1,
            stock_dimension * (stock_dimension + feature_dimension),
        ),
        torch.FloatTensor(actions).reshape(-1, stock_dimension),
    )
    interpolated_q = model.policy.evaluate_actions(
        torch.FloatTensor(interpolated_input).reshape(
            -1,
            stock_dimension * (stock_dimension + feature_dimension),
        ),
        torch.FloatTensor(actions).reshape(-1, stock_dimension),
    )
    forward_q = forward_q[0].detach().cpu().numpy()[0]
    interpolated_q = interpolated_q[0].detach().cpu().numpy()[0]
    return (forward_q - interpolated_q) / h


def calculate_saliency_maps(trade, unique_trade_date, drl_models, drl_results, app_config):
    tech_indicator_list = app_config.environment.tech_indicator_list
    stock_dimension = len(trade.tic.unique())
    feature_dimension = len(tech_indicator_list)
    meta_q = {"date": [], "feature": [], "Saliency Map": [], "algo": []}

    for algo in ["A2C", "PPO"]:
        prec_step = 1e-2 if algo == "A2C" else 1e-1
        model = drl_models[algo]
        df_actions = drl_results[algo]["actions"]

        for i in range(len(unique_trade_date) - 1):
            date = unique_trade_date[i]
            covs = trade[trade["date"] == date].cov_list.iloc[0]
            features = trade[trade["date"] == date][tech_indicator_list].values
            actions = df_actions.loc[date].values

            for feature_idx, feature_name in enumerate(tech_indicator_list):
                int_grad_per_feature = 0
                for stock_idx in range(features.shape[0]):
                    avg_interpolated_grad = 0
                    for alpha in range(1, 51):
                        scale = 1 / 50
                        baseline_features = copy.deepcopy(features)
                        baseline_features[:, feature_idx] = [0] * stock_dimension
                        interpolated_features = baseline_features + scale * alpha * (
                            features - baseline_features
                        )
                        interpolated_input = np.append(
                            covs,
                            interpolated_features.T,
                            axis=0,
                        )
                        interpolated_gradient = calculate_gradient(
                            model,
                            interpolated_input,
                            actions,
                            feature_idx,
                            stock_idx,
                            stock_dimension,
                            feature_dimension,
                            h=prec_step,
                        )[0]
                        avg_interpolated_grad += interpolated_gradient * scale
                    int_grad_per_stock = (
                        features[stock_idx][feature_idx] * avg_interpolated_grad
                    )
                    int_grad_per_feature += int_grad_per_stock

                meta_q["date"].append(date)
                meta_q["algo"].append(algo)
                meta_q["feature"].append(feature_name)
                meta_q["Saliency Map"].append(int_grad_per_feature)

    return pd.DataFrame(meta_q)


def calculate_meta_score_coefficients(
    df,
    unique_trade_date,
    ml_predictions,
    drl_results,
    app_config,
):
    tech_indicator_list = app_config.environment.tech_indicator_list
    meta_score_coef = {"date": [], "coef": [], "algo": []}
    weights_map = {
        "LR": ml_predictions["LR"][3],
        "RF": ml_predictions["RF"][3],
        "DT": ml_predictions["DT"][3],
        "SVM": ml_predictions["SVM"][3],
        "A2C": drl_results["A2C"]["weights"],
        "PPO": drl_results["PPO"]["weights"],
        "Reference Model": ml_predictions["Reference Model"][3],
    }

    for algo, weights in weights_map.items():
        for i in range(len(unique_trade_date) - 1):
            date = unique_trade_date[i]
            next_date = unique_trade_date[i + 1]
            df_temp = df[df.date == date].reset_index(drop=True)
            df_temp_next = df[df.date == next_date].reset_index(drop=True)
            weight_piece = weights[weights.date == date].iloc[0]["weights"]
            piece_return = pd.DataFrame(
                df_temp_next.return_list.iloc[0].loc[next_date]
            ).reset_index()
            piece_return.columns = ["tic", "return"]
            x = df_temp[[*tech_indicator_list, "tic"]]
            piece = weight_piece.merge(x, on="tic").merge(piece_return, on="tic")
            piece["Y"] = piece["return"] * piece["weight"]
            x_ols = sm.add_constant(piece[tech_indicator_list])
            y_ols = piece[["Y"]]
            results = sm.OLS(y_ols, x_ols).fit()
            meta_score_coef["coef"].append((x_ols * results.params).sum(axis=0))
            meta_score_coef["date"].append(date)
            meta_score_coef["algo"].append(algo)

    return pd.DataFrame(meta_score_coef)


def calculate_performance_scores(
    unique_trade_date,
    meta_score_coef,
    meta_q,
    app_config,
):
    tech_indicator_list = app_config.environment.tech_indicator_list
    performance_score = {"date": [], "algo": [], "score": []}

    for date_ in unique_trade_date:
        if len(meta_score_coef[meta_score_coef["date"] == date_]) == 0:
            continue

        def get_coef(algo):
            return meta_score_coef[
                (meta_score_coef["date"] == date_) & (meta_score_coef["algo"] == algo)
            ]["coef"].values[0][tech_indicator_list].values

        reference_coef = get_coef("Reference Model")
        score_map = {
            "LR": np.corrcoef(get_coef("LR"), reference_coef)[0][1],
            "RF": np.corrcoef(get_coef("RF"), reference_coef)[0][1],
            "DT": np.corrcoef(get_coef("DT"), reference_coef)[0][1],
            "SVM": np.corrcoef(get_coef("SVM"), reference_coef)[0][1],
            "A2C": np.corrcoef(
                meta_q[(meta_q["date"] == date_) & (meta_q["algo"] == "A2C")][
                    "Saliency Map"
                ].values,
                reference_coef,
            )[0][1],
            "PPO": np.corrcoef(
                meta_q[(meta_q["date"] == date_) & (meta_q["algo"] == "PPO")][
                    "Saliency Map"
                ].values,
                reference_coef,
            )[0][1],
        }
        for algo, score in score_map.items():
            performance_score["date"].append(date_)
            performance_score["algo"].append(algo)
            performance_score["score"].append(score)

    return pd.DataFrame(performance_score)


def calculate_multi_performance_scores(
    unique_trade_date,
    meta_score_coef,
    meta_q,
    app_config,
):
    tech_indicator_list = app_config.environment.tech_indicator_list
    window = app_config.evaluation.multi_step_window
    multi_performance_score = {"date": [], "algo": [], "score": []}

    for i in range(len(unique_trade_date) - window):
        date_ = unique_trade_date[i]
        if len(meta_score_coef[meta_score_coef["date"] == date_]) == 0:
            continue

        def get_coef(algo, date_value=date_):
            return meta_score_coef[
                (meta_score_coef["date"] == date_value)
                & (meta_score_coef["algo"] == algo)
            ]["coef"].values[0][tech_indicator_list].values

        reference_coef = get_coef("Reference Model")
        for w in range(1, window):
            reference_coef += get_coef("Reference Model", unique_trade_date[i + w])
        reference_coef /= window

        score_map = {
            "LR": np.corrcoef(get_coef("LR"), reference_coef)[0][1],
            "RF": np.corrcoef(get_coef("RF"), reference_coef)[0][1],
            "DT": np.corrcoef(get_coef("DT"), reference_coef)[0][1],
            "SVM": np.corrcoef(get_coef("SVM"), reference_coef)[0][1],
            "A2C": np.corrcoef(
                meta_q[(meta_q["date"] == date_) & (meta_q["algo"] == "A2C")][
                    "Saliency Map"
                ].values,
                reference_coef,
            )[0][1],
            "PPO": np.corrcoef(
                meta_q[(meta_q["date"] == date_) & (meta_q["algo"] == "PPO")][
                    "Saliency Map"
                ].values,
                reference_coef,
            )[0][1],
        }
        for algo, score in score_map.items():
            multi_performance_score["date"].append(date_)
            multi_performance_score["algo"].append(algo)
            multi_performance_score["score"].append(score)

    return pd.DataFrame(multi_performance_score)


def build_meta_score(ml_predictions, drl_results, baseline_stats):
    meta_score = {
        "Annual return": [],
        "Annual volatility": [],
        "Max drawdown": [],
        "Sharpe ratio": [],
        "Algorithm": [],
        "Calmar ratio": [],
    }
    stats_map = {
        "DT": ml_predictions["DT"][1],
        "LR": ml_predictions["LR"][1],
        "SVM": ml_predictions["SVM"][1],
        "RF": ml_predictions["RF"][1],
        "Reference Model": ml_predictions["Reference Model"][1],
        "PPO": drl_results["PPO"]["stats"],
        "DJI": baseline_stats,
        "A2C": drl_results["A2C"]["stats"],
    }

    for name, stats in stats_map.items():
        meta_score["Algorithm"].append(name)
        meta_score["Annual return"].append(stats["Annual return"])
        meta_score["Annual volatility"].append(stats["Annual volatility"])
        meta_score["Max drawdown"].append(stats["Max drawdown"])
        meta_score["Sharpe ratio"].append(stats["Sharpe ratio"])
        meta_score["Calmar ratio"].append(stats["Calmar ratio"])

    return pd.DataFrame(meta_score).sort_values("Sharpe ratio")


def build_positive_ratios(performance_score, multi_performance_score, meta_score):
    positive_ratio = pd.DataFrame(
        performance_score.groupby("algo").apply(lambda x: np.mean(x["score"]))
    ).reset_index()
    positive_ratio.columns = ["algo", "avg_correlation_coefficient"]
    positive_ratio["Sharpe Ratio"] = 0.0

    positive_ratio_multi = pd.DataFrame(
        multi_performance_score.groupby("algo").apply(lambda x: np.mean(x["score"]))
    ).reset_index()
    positive_ratio_multi.columns = ["algo", "avg_correlation_coefficient"]
    positive_ratio_multi["Sharpe Ratio"] = 0.0

    for algo in ["A2C", "PPO", "LR", "DT", "RF", "SVM"]:
        sharpe_ratio = meta_score.loc[
            meta_score["Algorithm"] == algo,
            "Sharpe ratio",
        ].values[0]
        positive_ratio.loc[
            positive_ratio["algo"] == algo,
            "Sharpe Ratio",
        ] = sharpe_ratio
        positive_ratio_multi.loc[
            positive_ratio_multi["algo"] == algo,
            "Sharpe Ratio",
        ] = sharpe_ratio

    positive_ratio.sort_values("Sharpe Ratio", inplace=True)
    positive_ratio_multi.sort_values("Sharpe Ratio", inplace=True)
    return positive_ratio, positive_ratio_multi
