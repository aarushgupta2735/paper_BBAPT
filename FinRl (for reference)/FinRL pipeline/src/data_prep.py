import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


def setup_directories(directory_config):
    for d in [
        directory_config.DATA_SAVE_DIR,
        directory_config.TRAINED_MODEL_DIR,
        directory_config.TENSORBOARD_LOG_DIR,
        directory_config.RESULTS_DIR,
    ]:
        os.makedirs("./" + d, exist_ok=True)


def data_prep(app_config):
    df = YahooDownloader(
        start_date=app_config.data_prep.start_date,
        end_date=app_config.data_prep.end_date,
        ticker_list=app_config.data_prep.ticker_list,
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=app_config.data_prep.use_technical_indicator,
        use_turbulence=app_config.data_prep.use_turbulence,
        user_defined_feature=app_config.data_prep.user_defined_feature,
    )
    df = fe.preprocess_data(df)

    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []
    lookback = app_config.data_prep.lookback

    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(index="date", columns="tic", values="close")
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame({
        "date": df.date.unique()[lookback:],
        "cov_list": cov_list,
        "return_list": return_list,
    })
    df = df.merge(df_cov, on="date")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    return df
    
