"""Solver calls shared by the experiment scripts and notebooks."""

import numpy as np

from badr.algorithms import SLSQP
from badr.oracles import ImplicitOracle


def solve_oracle(oracle, max_iter=500):
    """Run SLSQP and return its weights; fail explicitly if it does not converge."""
    # Differentiate the actual fitted-model objective, including preprocessing.
    solver = SLSQP(use_oracle_gradient=False)
    solver.set_oracle(oracle)
    solver.run(max_iter=max_iter, verbose=0)
    if not solver.success:
        raise RuntimeError(f"SLSQP failed: {solver.message}")
    weights = np.asarray(solver.group_weights, dtype=float)
    if (
        weights.shape != (oracle.n_groups,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < -1e-7)
        or not np.isclose(weights.sum(), 1.0, atol=1e-6, rtol=0)
    ):
        raise RuntimeError("SLSQP returned invalid simplex weights")
    return weights


def solve_weights(model, metric, dset, max_iter=500):
    """Optimize training fairness through the lower-level model using SLSQP."""
    metric.set_model(model)
    oracle = ImplicitOracle(dset, model, metric, train_test="train")
    return solve_oracle(oracle, max_iter=max_iter)
