#!/usr/bin/env python3
"""Generate Stage-1 accuracy summary and Stage-2 runtime tables from the raw CSV.

Reads experiments/results/raw_results.csv and writes:

  experiments/results/summary_accuracy.md     (Stage 1: validation-selected test acc)
  experiments/results/summary_solver_time.md  (Stage 2: solver-only timing)
  experiments/results/summary_endtoend.md     (Stage 2: end-to-end runtime decomposition)

The selection rule is: for each (dataset, solver), pick the configuration with
the highest mean validation accuracy across seeds, then report the corresponding
mean test accuracy and timings. RASSG-r configurations include their inner-loop
budget. Hyperparameters are never selected on the test set.

Usage:
    PYTHONPATH=src uv run python experiments/summarize_results.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CSV = RESULTS_DIR / "raw_results.csv"
PRIMARY_SOLVER = "RASSG-r"
LEGACY_PRIMARY_SOLVER = "ASSG-r"
PRIMARY_SOLVERS = (PRIMARY_SOLVER, LEGACY_PRIMARY_SOLVER)
PRIMARY_SOLVER_LABEL = "Snacks"


def _display_solver(solver: str) -> str:
    if solver in PRIMARY_SOLVERS:
        return PRIMARY_SOLVER_LABEL
    return solver


def _display_solvers(raw_solvers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for solver in raw_solvers:
        name = _display_solver(solver)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _config_key(row: dict) -> tuple:
    """Return the hyperparameter key used for validation selection."""
    key = [row["dataset"], row["solver"], row.get("lam", ""), row.get("m", "")]
    if row["solver"] in PRIMARY_SOLVERS:
        key.extend(
            [
                row.get("n_stages", ""),
                row.get("m_inner", ""),
                row.get("alpha", ""),
                row.get("q", ""),
                row.get("adaptive_decay", ""),
            ]
        )
    return tuple(key)


def _group_by_config(rows: list[dict]):
    """Group rows by solver configuration; return dict of lists."""
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[_config_key(r)].append(r)
    return g


def _select_best(rows: list[dict]) -> dict:
    """For each (dataset, solver), pick the config with max mean val_acc.

    Returns: {(dataset, solver): {lam, m, val_acc, test_acc, train_time_solver,
                                   total_train_time, total_time, n_seeds, ...}}
    """
    by_cfg = _group_by_config(rows)
    by_pair: dict[tuple, list[tuple]] = defaultdict(list)
    for key, recs in by_cfg.items():
        ds, solver, lam, m = key[:4]
        vals = [_f(r.get("val_acc")) for r in recs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        tests = [_f(r.get("test_acc")) for r in recs]
        tests = [v for v in tests if v is not None]
        train_t = [_f(r.get("train_time_solver")) for r in recs]
        train_t = [v for v in train_t if v is not None]
        total_train_t = [_f(r.get("total_train_time")) for r in recs]
        total_train_t = [v for v in total_train_t if v is not None]
        total_t = [_f(r.get("total_time")) for r in recs]
        total_t = [v for v in total_t if v is not None]
        nys_fit = [_f(r.get("nystrom_fit_time")) for r in recs]
        nys_fit = [v for v in nys_fit if v is not None]
        nys_tr = [_f(r.get("nystrom_transform_train_time")) for r in recs]
        nys_tr = [v for v in nys_tr if v is not None]
        pred_t = [_f(r.get("predict_time_solver")) for r in recs]
        pred_t = [v for v in pred_t if v is not None]
        by_pair[(ds, solver)].append(
            {
                "lam": lam,
                "m": m,
                "m_inner": recs[0].get("m_inner", ""),
                "val_acc": mean(vals),
                "test_acc": mean(tests) if tests else None,
                "train_time_solver": mean(train_t) if train_t else None,
                "total_train_time": mean(total_train_t) if total_train_t else None,
                "total_time": mean(total_t) if total_t else None,
                "nystrom_fit_time": mean(nys_fit) if nys_fit else None,
                "nystrom_transform_train_time": mean(nys_tr) if nys_tr else None,
                "predict_time_solver": mean(pred_t) if pred_t else None,
                "n_seeds": len(recs),
            }
        )

    out: dict[tuple, dict] = {}
    for key, candidates in by_pair.items():
        out[key] = max(candidates, key=lambda c: c["val_acc"])
    return out


def _fmt(v, fmt=".4f"):
    if v is None:
        return "NA"
    return format(v, fmt)


def stage1_accuracy_table(best: dict, out_path: Path):
    """Stage 1 main table:
    dataset | best method | best test acc | Snacks test acc | Snacks gap | Snacks solver time
    Plus a per-method test-accuracy comparison.
    """
    datasets = sorted({k[0] for k in best})
    raw_solvers = sorted({k[1] for k in best})
    solvers = _display_solvers(raw_solvers)

    lines: list[str] = []
    lines.append("# Stage 1 — Accuracy summary (validation-selected hyperparameters)\n")
    lines.append(
        "Test accuracies are means across seeds at the configuration with highest mean validation accuracy.\n"
    )

    # Per-method accuracy table.
    header = ["dataset"] + solvers
    lines.append("## Test accuracy by method\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for ds in datasets:
        row = [ds]
        for display_solver in solvers:
            if display_solver == PRIMARY_SOLVER_LABEL:
                entry = best.get((ds, PRIMARY_SOLVER)) or best.get(
                    (ds, LEGACY_PRIMARY_SOLVER)
                )
            else:
                entry = best.get((ds, display_solver))
            row.append(_fmt(entry["test_acc"]) if entry else "NA")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # RASSG-r gap to best.
    lines.append("## Snacks gap to best baseline\n")
    lines.append(
        "| dataset | best method | best test acc | Snacks test acc | gap | Snacks solver time (s) | Snacks total time (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for ds in datasets:
        ds_best = None
        ds_best_solver = None
        for raw_solver in raw_solvers:
            e = best.get((ds, raw_solver))
            if e is None or e["test_acc"] is None:
                continue
            if ds_best is None or e["test_acc"] > ds_best:
                ds_best = e["test_acc"]
                ds_best_solver = _display_solver(raw_solver)
        primary = best.get((ds, PRIMARY_SOLVER)) or best.get(
            (ds, LEGACY_PRIMARY_SOLVER)
        )
        if primary is None or primary["test_acc"] is None:
            continue
        gap = ds_best - primary["test_acc"] if ds_best is not None else None
        lines.append(
            f"| {ds} | {ds_best_solver} | {_fmt(ds_best)} | {_fmt(primary['test_acc'])} | "
            f"{_fmt(gap)} | {_fmt(primary['train_time_solver'], '.3f')} | {_fmt(primary['total_time'], '.3f')} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def stage2_solver_time_table(best: dict, out_path: Path):
    """Stage 2 — solver-only timing (excludes Nyström construction).

    sklearn-Nystrom is excluded here because its end-to-end pipeline does not
    expose a comparable solver-only time.
    """
    datasets = sorted({k[0] for k in best})
    raw_solvers = sorted({k[1] for k in best if k[1] != "sklearn-Nystrom"})
    solvers = _display_solvers(raw_solvers)

    lines: list[str] = []
    lines.append("# Stage 2 — Solver-only training time (seconds)\n")
    lines.append("Time to fit the linear model on already-computed Nyström features.")
    lines.append(
        "Excludes bandwidth selection, column selection, Nyström fit/transform, and prediction."
    )
    lines.append("Hyperparameters selected by mean validation accuracy across seeds.")
    lines.append(
        "sklearn-Nystrom omitted: its end-to-end pipeline has no comparable solver-only split.\n"
    )

    lines.append("| dataset | " + " | ".join(solvers) + " |")
    lines.append("|" + "|".join(["---"] * (len(solvers) + 1)) + "|")
    for ds in datasets:
        row = [ds]
        for display_solver in solvers:
            if display_solver == PRIMARY_SOLVER_LABEL:
                e = best.get((ds, PRIMARY_SOLVER)) or best.get(
                    (ds, LEGACY_PRIMARY_SOLVER)
                )
            else:
                e = best.get((ds, display_solver))
            row.append(_fmt(e["train_time_solver"], ".4f") if e else "NA")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def stage2_endtoend_table(best: dict, out_path: Path):
    """Stage 2 — end-to-end runtime decomposition.

    Reports (mean over seeds, at validation-best (lam, m)):
      nystrom_fit_time, nystrom_transform_train_time, train_time_solver,
      predict_time_solver, total_time.
    For sklearn-Nystrom only total_time is meaningful; per-step decomposition is NA.
    """
    datasets = sorted({k[0] for k in best})
    raw_solvers = sorted({k[1] for k in best})
    solvers = _display_solvers(raw_solvers)

    lines: list[str] = []
    lines.append("# Stage 2 — End-to-end runtime decomposition (seconds)\n")
    lines.append("Means across seeds at validation-selected (lam, m). Columns:")
    lines.append("`nys_fit` = Nyström column eigendecomposition;")
    lines.append("`nys_transform_train` = transform of training data;")
    lines.append("`solver_fit` = solver training on Nyström features;")
    lines.append("`predict` = prediction on already-embedded test features;")
    lines.append(
        "`total` = full pipeline time (fit columns, transform train/val/test, solver fit, predict)."
    )
    lines.append(
        "sklearn-Nystrom reports only `total` because feature construction is bundled inside its estimator path.\n"
    )

    cols = ["nys_fit", "nys_transform_train", "solver_fit", "predict", "total"]
    lines.append("| dataset | solver | " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(cols) + 2)) + "|")
    for ds in datasets:
        for display_solver in solvers:
            if display_solver == PRIMARY_SOLVER_LABEL:
                e = best.get((ds, PRIMARY_SOLVER)) or best.get(
                    (ds, LEGACY_PRIMARY_SOLVER)
                )
            else:
                e = best.get((ds, display_solver))
            if e is None:
                continue
            row = [
                ds,
                display_solver,
                _fmt(e["nystrom_fit_time"], ".4f"),
                _fmt(e["nystrom_transform_train_time"], ".4f"),
                _fmt(e["train_time_solver"], ".4f"),
                _fmt(e["predict_time_solver"], ".4f"),
                _fmt(e["total_time"], ".4f"),
            ]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load(csv_path)
    if not rows:
        print(f"No rows in {csv_path}")
        return

    best = _select_best(rows)
    print(
        f"Loaded {len(rows)} rows; {len(best)} (dataset, solver) pairs after validation selection."
    )

    stage1_accuracy_table(best, out_dir / "summary_accuracy.md")
    stage2_solver_time_table(best, out_dir / "summary_solver_time.md")
    stage2_endtoend_table(best, out_dir / "summary_endtoend.md")


if __name__ == "__main__":
    main()
