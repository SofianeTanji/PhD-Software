#!/usr/bin/env python3
"""Generate LaTeX tables from raw_results.csv.

One table per dataset. Columns: solver time, train C-err, test C-err,
all shown as mean ± std across seeds. Best value per column is bolded.

Usage:
    PYTHONPATH=src uv run python experiments/generate_tables.py
    PYTHONPATH=src uv run python experiments/generate_tables.py --out experiments/results/tables.tex
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev as _stdev

RESULTS_CSV = Path(__file__).parent / "results" / "raw_results.csv"

SOLVER_DISPLAY = {
    "RASSG-r": "Snacks",
    "ASSG-r": "Snacks",
    "LibLinear": "LibLinear",
    "Pegasos": "Pegasos",
    "sklearn-Nystrom": "sklearn-Nystrom",
}

SOLVER_ORDER = ["LibLinear", "Pegasos", "sklearn-Nystrom", "RASSG-r", "ASSG-r"]


def _f(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _std(xs: list[float]) -> float:
    return _stdev(xs) if len(xs) > 1 else 0.0


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


def _fmt_time(mu: float, sd: float) -> str:
    return f"{mu:.1f} $\\pm$ {sd:.1f} s"


def _fmt_pct(mu: float, sd: float) -> str:
    return f"{100 * mu:.1f} $\\pm$ {100 * sd:.1f} \\%"


def load_and_aggregate(path: Path) -> dict[str, dict[str, dict]]:
    """For each (dataset, solver), return stats at the config with best mean val_acc."""
    with open(path) as f:
        rows = list(csv.DictReader(f))

    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r["dataset"], r["solver"], r["lam"], r["m"])].append(r)

    by_ds_solver: dict[tuple, list[dict]] = defaultdict(list)
    for (ds, solver, lam, m), recs in by_cfg.items():
        vals = [v for r in recs if (v := _f(r.get("val_acc"))) is not None]
        if not vals:
            continue
        entry: dict = {"lam": lam, "m": m, "val_acc": mean(vals)}
        for field in ("train_acc", "test_acc", "total_train_time", "nystrom_fit_time"):
            xs = [v for r in recs if (v := _f(r.get(field))) is not None]
            entry[field] = mean(xs) if xs else None
            entry[field + "_std"] = _std(xs) if xs else 0.0
        by_ds_solver[(ds, solver)].append(entry)

    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for (ds, solver), cands in by_ds_solver.items():
        result[ds][solver] = max(cands, key=lambda c: c["val_acc"])
    return result


def _best(solver_stats: dict[str, dict], field: str, sign: float = 1.0) -> float | None:
    """Return the best (sign=1 → highest, sign=-1 → lowest) mean value across solvers."""
    vals = [sign * s[field] for s in solver_stats.values() if s.get(field) is not None]
    return sign * max(vals) if vals else None


def make_table(dataset: str, solver_stats: dict[str, dict]) -> str:
    ref = (
        solver_stats.get("RASSG-r")
        or solver_stats.get("ASSG-r")
        or next(iter(solver_stats.values()))
    )
    m = ref["m"]
    nys_time = ref.get("nystrom_fit_time")
    nys_str = f"{nys_time:.1f}s" if nys_time is not None else "?"

    test_accs = [
        s["test_acc"] for s in solver_stats.values() if s.get("test_acc") is not None
    ]
    optimal_cerr = min(1 - a for a in test_accs) if test_accs else None
    optimal_str = f"{100 * optimal_cerr:.1f} \\%" if optimal_cerr is not None else "?"

    best_time = _best(solver_stats, "total_train_time", sign=-1)  # lowest
    best_train_acc = _best(
        solver_stats, "train_acc", sign=1
    )  # highest acc = lowest C-err
    best_test_acc = _best(solver_stats, "test_acc", sign=1)

    EPS = 1e-4

    lines = [
        r"\begin{table}[htbp!]",
        r"\centering",
        f"\\caption{{{dataset}, $m = {m}$. Nystrom features precomputed in ${nys_str}$}}",
        r"\begin{tabular}{@{}rccc@{}}",
        r"\toprule",
        (
            f"\\multicolumn{{1}}{{c}}{{\\textbf{{{dataset}}}}} & "
            f"Time (s) & "
            f"Train C-err & "
            f"Test C-err (optimal = {optimal_str}) \\\\ \\midrule"
        ),
    ]

    solvers = [s for s in SOLVER_ORDER if s in solver_stats]
    solvers += [s for s in solver_stats if s not in SOLVER_ORDER]

    for solver in solvers:
        s = solver_stats[solver]
        name = SOLVER_DISPLAY.get(solver, solver)

        # Time
        t_mu, t_sd = s.get("total_train_time"), s.get("total_train_time_std", 0.0)
        if t_mu is not None:
            cell_time = _fmt_time(t_mu, t_sd)
            if best_time is not None and abs(t_mu - best_time) < EPS:
                cell_time = _bold(cell_time)
        else:
            cell_time = "---"

        # Train C-err
        tr_mu, tr_sd = s.get("train_acc"), s.get("train_acc_std", 0.0)
        if tr_mu is not None:
            cell_train = _fmt_pct(1 - tr_mu, tr_sd)
            if best_train_acc is not None and abs(tr_mu - best_train_acc) < EPS:
                cell_train = _bold(cell_train)
        else:
            cell_train = "---"

        # Test C-err
        te_mu, te_sd = s.get("test_acc"), s.get("test_acc_std", 0.0)
        if te_mu is not None:
            cell_test = _fmt_pct(1 - te_mu, te_sd)
            if best_test_acc is not None and abs(te_mu - best_test_acc) < EPS:
                cell_test = _bold(cell_test)
        else:
            cell_test = "---"

        lines.append(f"{name} & {cell_time} & {cell_train} & {cell_test} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", default=str(RESULTS_CSV))
    p.add_argument("--out", default="experiments/results/tables.tex")
    args = p.parse_args()

    data = load_and_aggregate(Path(args.csv))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tables = []
    for dataset in sorted(data):
        tables.append(make_table(dataset, data[dataset]))
        print(f"  {dataset}: {sorted(data[dataset])}")

    out_path.write_text("\n\n".join(tables) + "\n")
    print(f"\nWrote {len(tables)} tables to {out_path}")


if __name__ == "__main__":
    main()
