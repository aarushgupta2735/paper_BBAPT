"""
Behavioral hyperparameter grid search.

Per the paper's "Calibration of behavioral hyperparameters" section: each
behavioral profile (loss-aversion, overconfidence) is governed by a small
set of interpretable parameters. The paper trains a fresh agent per
combination and picks whichever configuration maximizes average episodic
reward on a validation split.

DEVIATION FROM THE PAPER (by explicit choice, discussed before writing this):
we do NOT retrain the RL agent for every combination. We train ONE neutral
agent once, freeze it, and only vary the behavioral post-processing layer
(apply_behavioural_mapping) on top of its fixed outputs. This is much
cheaper and is a legitimate thing to do since, per the paper's own Eq. (15),
behavioral adjustments are a stateless function of (w(0), unrealized
returns, params) applied AFTER the policy acts - they don't change what the
policy would have chosen. It does mean we're not asking "what's the best
(policy, behavioral params) pair jointly" - just "given this fixed policy,
what's the best behavioral params". If you want the paper's exact
methodology later, swap `_evaluate_config`'s TODO to train a fresh agent.

Validation split: last ~20% of the existing train window (config's
train_starting_date -> train_ending_date), matching the ~80/20 ratio used
in the paper's Table 4 regime splits.

Selection metric: info["sharpe_ratio"] as tracked by PortfolioEnv itself -
an EXPANDING Sharpe accumulated over self.profits since env.reset(). By the
final step of a full walk-forward episode this equals the whole-period
Sharpe (Eq. 16/24), so we simply read it off the last info dict rather than
reconstructing it here. This is intentionally the SAME env attribute used
for reporting/comparison elsewhere, NOT info["rolling_sharpe_ratio"] (the
fixed-window Sharpe used as the dense training reward inside step()) -
using the rolling one here would score configs on an arbitrary tail window
instead of the full validation period.
"""

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd

from BBAPT.src.data_prep import data_prep, data_for_timesnet
from BBAPT.src.portfolio_env import PortfolioEnv
from BBAPT.src.behavioral_map import apply_behavioural_mapping


def make_timesnet_forecast_fn(nf):
    """
    Build a forecast_fn backed by an ALREADY-FITTED NeuralForecast object,
    calling it exactly the way main.py's inference loop does:

        forecast = nf.predict(df=data_for_timesnet(history_df))
        avg_return = forecast["TimesNet"].mean()

    `nf` must already be fit (nf.fit(...) called once, upstream, on the
    training data) - this function only ever calls .predict(), it never
    refits. That keeps every grid cell scored against the identical
    forecasting model, and avoids the cost of re-fitting TimesNet per
    combination.

    Walk-forward safety: `history_df` passed in by `_run_walk_forward` is
    already sliced to `val_df[val_df['ds'] <= date]` - i.e. only data up to
    and including the current date - so this function never sees future
    rows. It relies on the caller to keep passing correctly-sliced history;
    it does no slicing itself.
    """

    def forecast_fn(date, history_df):
        forecast = nf.predict(df=data_for_timesnet(history_df))
        return forecast["TimesNet"].mean()

    return forecast_fn


def make_validation_split(config, val_fraction: float = 0.20):
    """
    Carve the last `val_fraction` of the existing train window off as a
    validation window. Returns (train_start, train_end, val_start, val_end)
    as date strings, all still within the ORIGINAL train period - the real
    test period (config.test_starting_date/test_ending_date) is untouched.
    """
    full_start = pd.Timestamp(config.train_starting_date)
    full_end = pd.Timestamp(config.train_ending_date)
    total_days = (full_end - full_start).days

    val_days = int(round(total_days * val_fraction))
    val_start = full_end - pd.Timedelta(days=val_days)

    # inner-train end is the day before validation starts
    inner_train_end = val_start - pd.Timedelta(days=1)

    return (
        full_start.strftime("%Y-%m-%d"),
        inner_train_end.strftime("%Y-%m-%d"),
        val_start.strftime("%Y-%m-%d"),
        full_end.strftime("%Y-%m-%d"),
    )


def _run_walk_forward(model_rl, env, val_df, config, forecast_fn):
    """
    Run one frozen agent through one full walk-forward pass over `val_df`,
    applying the behavioral mapping at each step, and return the FINAL
    `info` dict once the episode ends.

    The env now tracks its own expanding, full-episode Sharpe ratio
    internally (info["sharpe_ratio"] - accumulated over `self.profits`
    since reset()), separate from info["rolling_sharpe_ratio"] (the
    windowed reward signal used for training). Since the expanding version
    is already a whole-episode statistic, by the last step of the episode
    it IS the full-period Sharpe (Eq. 16/24) - no need to reconstruct
    profits or recompute anything here; we just read it off the last info.

    `forecast_fn(date, data_so_far) -> avg_return` lets the caller plug in
    whatever forecast source it wants (see make_timesnet_forecast_fn above).
    """
    obs, info = env.reset()
    done = False
    date = info["date"]

    while not done:
        action, _ = model_rl.predict(obs, deterministic=True)
        avg_return = forecast_fn(date, val_df[val_df["ds"] <= date])
        action = apply_behavioural_mapping(action, avg_return, val_df, date, config)
        obs, reward, done, truncated, info = env.step(action)
        date = info["date"]

    return info


def _apply_param_overrides(config, param_dict):
    """Return a new config with the given field overrides applied."""
    return replace(config, **param_dict)


def run_grid_search(
    base_config,
    model_rl,
    forecast_fn,
    profile: str,
    param_grid: dict,
    val_fraction: float = 0.20,
):
    """
    Grid-search one behavioral profile's hyperparameters against a
    validation split, using a single frozen trained agent.

    Args:
        base_config: the appConfig instance used for training (unmodified).
        model_rl: a TRAINED, FROZEN stable-baselines3 model (e.g. A2C).
            It is only used for .predict() here - never retrained.
        forecast_fn: callable(date, history_df) -> avg_return, used in
            place of TimesNet during the search (see note below).
        profile: "loss_aversion" or "overconfidence" - which set of
            config fields to sweep.
        param_grid: dict mapping config field name -> list of candidate
            values, e.g. {"ra_upper_threshold": [...], "ra_k_gain": [...]}.
            Only fields relevant to the chosen profile should be passed;
            all other behavioral fields are held at base_config's values.
        val_fraction: fraction of the train window to carve off as
            validation (see make_validation_split).

    Returns:
        (best_config, results) where results is a list of
        (param_dict, val_sharpe) tuples for every combination tried,
        sorted best-first.

    NOTE on forecast_fn during search: re-running TimesNet training/
    inference inside every grid cell would be extremely slow (TimesNet
    training happens once, upstream, in main.py). Pass in a lightweight
    stand-in here, e.g. a function that returns the trailing-N-day mean
    return of the equally-weighted index as a proxy signal, OR reuse a
    single already-fitted `nf: NeuralForecast` object's .predict() if you
    want the exact same forecasts main.py would use at evaluation time.
    Either way, forecast_fn is called identically across every grid cell,
    so comparisons between cells remain fair.
    """
    if profile not in ("loss_aversion", "overconfidence"):
        raise ValueError(f"Unknown profile: {profile!r}")

    train_start, train_end, val_start, val_end = make_validation_split(
        base_config, val_fraction
    )
    val_df = data_prep(val_start, val_end, base_config.ticker_list)

    field_names = list(param_grid.keys())
    value_lists = [param_grid[f] for f in field_names]

    results = []
    for combo in itertools.product(*value_lists):
        param_dict = dict(zip(field_names, combo))
        trial_config = _apply_param_overrides(base_config, param_dict)

        env = PortfolioEnv(data=val_df, config=trial_config)
        final_info = _run_walk_forward(model_rl, env, val_df, trial_config, forecast_fn)
        val_sharpe = float(final_info["sharpe_ratio"])

        results.append((param_dict, val_sharpe))
        print(f"[{profile}] {param_dict} -> val_sharpe={val_sharpe:.4f}")

    results.sort(key=lambda r: r[1], reverse=True)
    best_params, best_sharpe = results[0]
    best_config = _apply_param_overrides(base_config, best_params)
    print(f"\n[{profile}] BEST: {best_params} -> val_sharpe={best_sharpe:.4f}")

    return best_config, results


# Default search ranges, taken directly from the paper's stated grid-search
# bounds ("Tuning the behavioral hyperparameters" section). Narrower than
# the paper's continuous ranges since these are discretized for a finite
# grid - widen/densify as compute allows.
LOSS_AVERSION_GRID = {
    "ra_lower_threshold": [-0.10, -0.06, -0.02],
    "ra_upper_threshold": [0.00, 0.05, 0.10],
    "ra_k_loss": [0.05, 0.20, 0.40],
    "ra_k_gain": [0.05, 0.20, 0.40],
    "ra_n": [0.60, 0.75, 0.90, 0.95],
}

OVERCONFIDENCE_GRID = {
    "oc_lower_threshold": [-0.12, -0.06, -0.01],
    "oc_upper_threshold": [0.01, 0.06, 0.12],
    "oc_k_loss": [0.05, 0.20, 0.40],
    "oc_k_gain": [0.05, 0.20, 0.40],
    "oc_n": [1.05, 1.20, 1.40],
}