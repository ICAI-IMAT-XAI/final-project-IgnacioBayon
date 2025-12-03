from itertools import product
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from utils.tscv import run_fixed_window_tscv


def hyperparameter_search_rf(
    X, y, dates, seq_lens, param_grid,
    train_size, test_size,
    verbose=True
):
    """
    Performs a manual grid search using your fixed-window TSCV evaluator.
    Returns best_params, best_seq_len, best_score.
    """

    best_score = float("inf")
    best_params = None
    best_seq_len = None

    # Generate all combinations of RF hyperparameters
    keys = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))

    print(f"Searching {len(combinations)} RF hyperparameter combinations...")
    print(f"Testing seq_lens: {seq_lens}")

    for seq_len in seq_lens:
        print(f"\n=== Testing seq_len = {seq_len} ===")

        for combo in combinations:
            params = dict(zip(keys, combo))
            params["random_state"] = 42
            params["n_jobs"] = -1

            if verbose:
                print(f"  → Trying params: {params}")

            # Run your TSCV evaluator
            mae_scores, rmse_scores = run_fixed_window_tscv(
                model_class=RandomForestRegressor,
                model_params=params,
                X=X,
                y=y,
                dates=dates,
                seq_len=seq_len,
                train_size=train_size,
                test_size=test_size,
                verbose=False
            )

            avg_mae = np.mean(mae_scores)

            if verbose:
                print(f"    MAE: {avg_mae:.5f}")

            # Check if best
            if avg_mae < best_score:
                best_score = avg_mae
                best_params = params
                best_seq_len = seq_len
                print(f"\tNew best: MAE={best_score:.5f}, seq_len={seq_len}, params={params}")

    return best_params, best_seq_len, best_score
