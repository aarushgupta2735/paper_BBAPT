import numpy as np
import pandas as pd
from finrl.plot import backtest_stats
from pypfopt import objective_functions, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def prepare_data(train_data, app_config):
    train_date = sorted(set(train_data.date.values))
    rows = []

    for i in range(len(train_date) - 1):
        current_date = train_date[i]
        next_date = train_date[i + 1]
        y = (
            train_data.loc[train_data["date"] == next_date]
            .return_list.iloc[0]
            .loc[next_date]
            .reset_index()
        )
        y.columns = ["tic", "return"]
        x = train_data.loc[train_data["date"] == current_date][
            ["tic", *app_config.environment.tech_indicator_list]
        ]
        train_piece = pd.merge(x, y, on="tic")
        train_piece["date"] = [current_date] * len(train_piece)
        rows.append(train_piece)

    train_data_ml = pd.concat(rows)
    x_out = train_data_ml[app_config.environment.tech_indicator_list].values
    y_out = train_data_ml[["return"]].values
    return x_out, y_out


def train_ml_models(train_data, app_config):
    train_x, train_y = prepare_data(train_data, app_config)
    y = train_y.reshape(-1)
    testing = app_config.testing

    return {
        "LR": LinearRegression().fit(train_x, train_y),
        "DT": DecisionTreeRegressor(
            random_state=testing.random_state,
            max_depth=testing.dt_max_depth,
            min_samples_split=testing.dt_min_samples_split,
        ).fit(train_x, y),
        "SVM": SVR(epsilon=testing.svm_epsilon).fit(train_x, y),
        "RF": RandomForestRegressor(
            max_depth=testing.rf_max_depth,
            min_samples_split=testing.rf_min_samples_split,
            random_state=testing.random_state,
        ).fit(train_x, y),
    }


def output_predict(df, unique_trade_date, model, app_config, reference_model=False):
    meta_coefficient = {"date": [], "weights": []}
    portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
    portfolio.loc[0, unique_trade_date[0]] = app_config.environment.initial_amount

    for i in range(len(unique_trade_date) - 1):
        current_date = unique_trade_date[i]
        next_date = unique_trade_date[i + 1]
        df_current = df[df.date == current_date].reset_index(drop=True)
        df_next = df[df.date == next_date].reset_index(drop=True)
        tics = df_current["tic"].values
        features = df_current[app_config.environment.tech_indicator_list].values

        if reference_model:
            mu = df_next.return_list[0].loc[next_date].values
            sigma = risk_models.sample_cov(df_next.return_list[0], returns_data=True)
        else:
            mu = model.predict(features)
            sigma = risk_models.sample_cov(df_current.return_list[0], returns_data=True)

        predicted_y_df = pd.DataFrame({
            "tic": tics.reshape(-1),
            "predicted_y": mu.reshape(-1),
        })
        ef = EfficientFrontier(mu, sigma)
        weights = ef.nonconvex_objective(
            objective_functions.sharpe_ratio,
            objective_args=(ef.expected_returns, ef.cov_matrix),
            weights_sum_to_one=True,
            constraints=[
                {"type": "ineq", "fun": lambda w: w},
                {"type": "ineq", "fun": lambda w: 1 - w},
            ],
        )

        weight_df = pd.DataFrame({
            "tic": list(weights.keys()),
            "weight": list(weights.values()),
        }).merge(predicted_y_df, on=["tic"])
        meta_coefficient["date"].append(current_date)
        meta_coefficient["weights"].append(weight_df)

        cap = portfolio.iloc[0, i]
        current_cash = [element * cap for element in list(weights.values())]
        current_shares = list(np.array(current_cash) / np.array(df_current.close))
        portfolio.iloc[0, i + 1] = np.dot(current_shares, np.array(df_next.close))

    portfolio = portfolio.T
    portfolio.columns = ["account_value"]
    portfolio = portfolio.reset_index()
    portfolio.columns = ["date", "account_value"]
    stats = backtest_stats(portfolio, value_col_name="account_value")
    portfolio_cumprod = (portfolio.account_value.pct_change() + 1).cumprod() - 1
    return portfolio, stats, portfolio_cumprod, pd.DataFrame(meta_coefficient)


def predict_ml_models(df, unique_trade_date, models, app_config):
    predictions = {}
    for name, model in models.items():
        predictions[name] = output_predict(df, unique_trade_date, model, app_config)
    predictions["Reference Model"] = output_predict(
        df, unique_trade_date, None, app_config, reference_model=True
    )
    return predictions
