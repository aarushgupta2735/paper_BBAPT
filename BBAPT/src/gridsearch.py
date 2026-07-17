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



def _run_walk_forward(model_rl, env, val_df, config, cv_df):
    """
    Run one frozen agent through one full walk-forward pass over `val_df`,
    applying the behavioral mapping at each step, and return the FINAL
    `info` dict once the episode ends.
    """
    obs, info = env.reset()
    done = False
    date = info["date"]

    while not done:
        action, _ = model_rl.predict(obs, deterministic=True)
        forecast = cv_df[cv_df["cutoff"] == date]
        avg_return = forecast["TimesNet"].mean()
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
    val_df,
    cv_df,
    profile: str,
    param_grid: dict,
):
    """
    Grid-search one behavioral profile's hyperparameters against a
    validation split, using a single frozen trained agent.
    """
    if profile not in ("ra", "oc"):
        raise ValueError(f"Unknown profile: {profile!r}")

    field_names = list(param_grid.keys())
    value_lists = [param_grid[f] for f in field_names]

    results = []
    for combo in itertools.product(*value_lists):
        param_dict = dict(zip(field_names, combo))
        trial_config = _apply_param_overrides(base_config, param_dict)

        env = PortfolioEnv(data=val_df, config=trial_config,already_weight=False)
        final_info = _run_walk_forward(model_rl, env, val_df, trial_config, cv_df)
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