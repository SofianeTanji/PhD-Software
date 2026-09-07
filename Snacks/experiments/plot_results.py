#!/usr/bin/env python3
"""Generate paper figures from experiment CSVs.

Usage:
    PYTHONPATH=src uv run python experiments/plot_results.py
    PYTHONPATH=src uv run python experiments/plot_results.py --csv experiments/results/raw_results.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev as _stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects
from matplotlib.colors import Normalize

from lib.plotstyle import apply_style, save_fig

# Apply custom matplotlib style + register BuRd colormap.
apply_style()

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CSV = RESULTS_DIR / "raw_results.csv"
DEFAULT_OUT = Path(__file__).parent / "figures"

LINEAR_SOLVERS = {"LibLinear", "Pegasos"}
PRIMARY_SOLVER = "RASSG-r"
LEGACY_PRIMARY_SOLVER = "ASSG-r"
PRIMARY_SOLVERS = (PRIMARY_SOLVER, LEGACY_PRIMARY_SOLVER)
PRIMARY_SOLVER_LABEL = "Snacks"

_SOLVER_COLOR = {
    "RASSG-r": "#194F8C",
    "ASSG-r": "#194F8C",
    "Pegasos": "#EB9486",
    "LibLinear": "#A0B78F",
    "sklearn-Nystrom": "#A53860",
}
TIME_PROFILE_ERROR_REL_TOL = 0.001


def _light_color(color: str, percent: float = 0.55) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    rgb = np.array(to_rgb(color))
    return tuple((1.0 - percent) * np.ones(3) + percent * rgb)


def _std(xs: list[float]) -> float:
    return _stdev(xs) if len(xs) > 1 else 0.0


def _ema(values: list[float], alpha: float = 0.25, min_len: int = 8) -> list[float]:
    if len(values) < min_len:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def _format_points(value: float | None) -> str:
    if value is None:
        return "unknown points"
    base, exp = f"{value:.0e}".split("e")
    return f"{base}e{int(exp)} points"


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _val_selected(records: list[dict]) -> dict[tuple, dict]:
    """For each (dataset, solver), return stats at the config with highest mean val_acc."""
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["dataset"], r["solver"], r.get("lam", ""), r.get("m", ""))
        by_cfg[key].append(r)

    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for (ds, s, lam, m), recs in by_cfg.items():
        vals = [_f(r.get("val_acc")) for r in recs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        entry = {"lam": lam, "m": m, "val_acc": mean(vals), "val_acc_std": _std(vals)}
        for field in (
            "train_acc",
            "test_acc",
            "train_time_solver",
            "total_time",
            "n_train",
            "n_val",
            "n_test",
        ):
            xs = [_f(r.get(field)) for r in recs]
            xs = [v for v in xs if v is not None]
            entry[field] = mean(xs) if xs else None
            entry[field + "_std"] = _std(xs) if xs else 0.0
        by_pair[(ds, s)].append(entry)

    return {
        key: max(cands, key=lambda c: c["val_acc"]) for key, cands in by_pair.items()
    }


def _primary_result(best: dict[tuple, dict], dataset: str) -> dict | None:
    for solver in PRIMARY_SOLVERS:
        if (dataset, solver) in best:
            return best[(dataset, solver)]
    return None


def _display_solver(solver: str) -> str:
    if solver in PRIMARY_SOLVERS:
        return PRIMARY_SOLVER_LABEL
    return solver


def _save(fig, out: Path, stem: str) -> None:
    save_fig(fig, out, stem)


def plot_accuracy_scatter(best: dict[tuple, dict], out: Path) -> None:
    """Scatter: Snacks test accuracy vs. best linear baseline, one point per dataset."""
    datasets = sorted({k[0] for k in best})
    xs, ys, labels = [], [], []
    for ds in datasets:
        primary = _primary_result(best, ds)
        if primary is None or primary["test_acc"] is None:
            continue
        linear_accs = [
            best[(ds, s)]["test_acc"]
            for s in LINEAR_SOLVERS
            if (ds, s) in best and best[(ds, s)]["test_acc"] is not None
        ]
        if not linear_accs:
            continue
        xs.append(max(linear_accs))
        ys.append(primary["test_acc"])
        labels.append(ds)

    if not xs:
        return

    lo = min(min(xs), min(ys)) - 0.01
    hi = max(max(xs), max(ys)) + 0.01

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--", zorder=0)
    ax.scatter(xs, ys, zorder=2)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7
        )
    ax.set_xlabel("Best linear baseline (test accuracy)")
    ax.set_ylabel(f"{PRIMARY_SOLVER_LABEL} (test accuracy)")
    ax.set_aspect("equal")
    _save(fig, out, "accuracy_scatter")
    plt.close(fig)


def _profile_solvers(best: dict[tuple, dict]) -> list[str]:
    return [
        s
        for s in ("RASSG-r", "ASSG-r", "Pegasos", "LibLinear", "sklearn-Nystrom")
        if any(k[1] == s for k in best)
    ]


def plot_test_error_profile(best: dict[tuple, dict], out: Path) -> None:
    """Dolan-Moré profile using test error ratios after validation selection."""
    solvers = _profile_solvers(best)
    datasets = sorted({k[0] for k in best})
    ratios_by_solver: dict[str, list[float]] = {s: [] for s in solvers}

    for ds in datasets:
        errs: dict[str, float] = {}
        for solver in solvers:
            row = best.get((ds, solver))
            acc = row.get("test_acc") if row else None
            if acc is None:
                continue
            errs[solver] = max(1e-12, 1.0 - acc)
        if len(errs) < 2:
            continue

        best_err = min(errs.values())
        for solver in solvers:
            if solver in errs:
                ratios_by_solver[solver].append(errs[solver] / best_err)

    _plot_profile(
        ratios_by_solver,
        out,
        stem="test_error_profile",
        xlabel="Test-error ratio to best solver",
        xscale="linear",
    )


def _plot_profile(
    ratios_by_solver: dict[str, list[float]],
    out: Path,
    *,
    stem: str,
    xlabel: str,
    xscale: str,
    ylabel: str = "Fraction of datasets",
    empty_note: str | None = None,
) -> None:
    """Plot a performance profile from already-computed per-solver ratios."""
    ratios_by_solver = {s: rs for s, rs in ratios_by_solver.items() if rs}
    if not ratios_by_solver:
        if empty_note is not None:
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            ax.axis("off")
            ax.text(0.5, 0.5, empty_note, ha="center", va="center")
            _save(fig, out, stem)
            plt.close(fig)
        return

    max_ratio = max(max(rs) for rs in ratios_by_solver.values())
    tau_max = max(1.05, max_ratio * 1.05)
    taus = np.unique(np.concatenate(([1.0], np.geomspace(1.01, tau_max, 200))))

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for solver, ratios in ratios_by_solver.items():
        color = _SOLVER_COLOR.get(solver, None)
        ys = [sum(r <= tau for r in ratios) / len(ratios) for tau in taus]
        ax.step(taus, ys, where="post", label=_display_solver(solver), color=color)

    ax.set_xscale(xscale)
    ax.set_xlim(1.0, tau_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale == "linear":
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.2g"))
    else:
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, out, stem)
    plt.close(fig)


def plot_performance_profile(best: dict[tuple, dict], out: Path) -> None:
    """Backward-compatible wrapper for the renamed test-error profile."""
    plot_test_error_profile(best, out)


def plot_scaling(scaling_records: list[dict], out: Path) -> None:
    """Solver training time vs m, one line per solver with ±1 stdev band."""
    by_solver_m: dict[tuple, list[float]] = defaultdict(list)
    for r in scaling_records:
        solver = r.get("solver", "")
        m = _f(r.get("m"))
        t = _f(r.get("solver_train_time"))
        if solver and m is not None and t is not None:
            by_solver_m[(solver, int(m))].append(t)

    solvers = sorted({s for s, _ in by_solver_m})
    if not solvers:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    for solver in solvers:
        color = _SOLVER_COLOR.get(solver, None)
        ms = sorted({m for s, m in by_solver_m if s == solver})
        ys_mean = [mean(by_solver_m[(solver, m)]) for m in ms]
        ys_std = [_std(by_solver_m[(solver, m)]) for m in ms]
        # Log scales require strictly positive bounds.
        ys_lo = [max(1e-12, mu - s) for mu, s in zip(ys_mean, ys_std)]
        ys_hi = [mu + s for mu, s in zip(ys_mean, ys_std)]
        ax.plot(ms, ys_mean, marker="o", ms=4, label=solver, color=color)
        ax.fill_between(ms, ys_lo, ys_hi, alpha=0.2, color=color)

    ax.set_xlabel("Number of Nyström columns $m$")
    ax.set_ylabel("Solver training time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
    ax.legend(fontsize=8)
    _save(fig, out, "scaling")
    plt.close(fig)


def plot_convergence(diag_records: list[dict], out: Path) -> None:
    """Train/test accuracy vs cumulative solver time, with ±1 stdev fill_between band."""
    datasets = sorted({r["dataset"] for r in diag_records})
    solvers = sorted({r.get("solver", "") for r in diag_records if r.get("solver")})
    if not datasets or not solvers:
        return

    ncols = min(2, len(datasets))
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.1 * ncols, 3.0 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    legend_items: dict[str, object] = {}

    for idx, ds in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        ds_recs = [r for r in diag_records if r["dataset"] == ds]
        plotted_xs: list[float] = []

        for solver in solvers:
            color = _SOLVER_COLOR.get(solver, None)
            line_color = _light_color(color) if color is not None else None
            sol_recs = [r for r in ds_recs if r.get("solver") == solver]
            if not sol_recs:
                continue

            # Stage timings vary by seed; align runs by stage and use mean
            # cumulative time as the x-coordinate.
            by_run: dict[tuple, list[dict]] = defaultdict(list)
            for r in sol_recs:
                by_run[(r.get("seed", ""), r.get("lam", ""), r.get("m", ""))].append(r)

            by_stage: dict[int, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for run_recs in by_run.values():
                elapsed = 0.0
                ordered_recs = sorted(
                    run_recs,
                    key=lambda r: (
                        _f(r.get("stage"))
                        if _f(r.get("stage")) is not None
                        else float("inf"),
                        _f(r.get("grad_evals"))
                        if _f(r.get("grad_evals")) is not None
                        else float("inf"),
                    ),
                )
                for r in ordered_recs:
                    dt = _f(r.get("stage_time"))
                    stage = _f(r.get("stage"))
                    if dt is None or stage is None:
                        continue
                    elapsed += max(0.0, dt)
                    if elapsed <= 0:
                        continue
                    by_stage[int(stage)]["time"].append(elapsed)
                    for metric in ("train_acc", "test_acc"):
                        v = _f(r.get(metric))
                        if v is not None:
                            by_stage[int(stage)][metric].append(v)

            stages = sorted(by_stage)
            if not stages:
                continue
            for metric, ls, metric_label in [
                ("train_acc", "-", "train"),
                ("test_acc", (0, (4, 2)), "test"),
            ]:
                xs_plotted, ys_mean, ys_lo, ys_hi = [], [], [], []
                for stage in stages:
                    vals = by_stage[stage][metric]
                    times = by_stage[stage]["time"]
                    if not vals or not times:
                        continue
                    errs = [1.0 - v for v in vals]
                    mu = mean(errs)
                    s = _std(errs)
                    xs_plotted.append(mean(times))
                    ys_mean.append(mu)
                    ys_lo.append(max(0.0, mu - s))
                    ys_hi.append(min(1.0, mu + s))
                if not ys_mean:
                    continue
                ys_mean = _ema(ys_mean)
                ys_lo = _ema(ys_lo)
                ys_hi = _ema(ys_hi)
                label = f"{_display_solver(solver)} ({metric_label})"
                is_test = metric == "test_acc"
                plot_color = color if is_test else line_color
                line_effects = (
                    [
                        path_effects.Stroke(linewidth=3.4, foreground="white"),
                        path_effects.Normal(),
                    ]
                    if is_test
                    else [
                        path_effects.Stroke(linewidth=3.0, foreground=color),
                        path_effects.Normal(),
                    ]
                )
                (line,) = ax.plot(
                    xs_plotted,
                    ys_mean,
                    ls=ls,
                    color=plot_color,
                    linewidth=2.0 if is_test else 1.8,
                    label=label,
                    zorder=8 if is_test else 6,
                    path_effects=line_effects if color is not None else None,
                )
                legend_items.setdefault(label, line)
                ax.fill_between(
                    xs_plotted,
                    ys_lo,
                    ys_hi,
                    alpha=0.06 if is_test else 0.08,
                    color=color,
                    zorder=2 if is_test else 1,
                )
                plotted_xs.extend(xs_plotted)

        # Log-scale solver time makes the fast early convergence visible.
        ax.set_xscale("log")
        if plotted_xs:
            ax.set_xlim(max(1e-9, min(plotted_xs) / 1.2), max(plotted_xs) * 1.2)

        ax.xaxis.set_major_locator(
            plt.matplotlib.ticker.LogLocator(base=10, numticks=4)
        )
        ax.xaxis.set_minor_locator(
            plt.matplotlib.ticker.LogLocator(base=10, subs=(2, 5), numticks=8)
        )
        ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
        ax.grid(True, which="major", color="#cfcfcf", linestyle=":", linewidth=0.6)
        ax.grid(True, which="minor", color="#dedede", linestyle=":", linewidth=0.35)
        ax.tick_params(which="both", width=1.2)
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.margins(y=0.08)
        n_train_values = [
            value for r in ds_recs if (value := _f(r.get("n_train"))) is not None
        ]
        n_train = mean(n_train_values) if n_train_values else None
        ax.set_title(f"{ds} ({_format_points(n_train)})")

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    bottom = 0.15 if nrows > 1 else 0.32
    xlabel_y = 0.075 if nrows > 1 else 0.18
    fig.subplots_adjust(
        left=0.11, right=0.995, top=0.94, bottom=bottom, wspace=0.28, hspace=0.48
    )
    fig.supxlabel("Elapsed solver time (s, log scale)", y=xlabel_y)
    fig.supylabel("Classification error (C-err)", x=0.005)
    if legend_items:
        preferred_order = [
            "Pegasos (train)",
            "Pegasos (test)",
            f"{PRIMARY_SOLVER_LABEL} (train)",
            f"{PRIMARY_SOLVER_LABEL} (test)",
        ]
        labels = [label for label in preferred_order if label in legend_items]
        labels.extend(label for label in legend_items if label not in labels)
        fig.legend(
            [legend_items[label] for label in labels],
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            fontsize=8,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor="#cfcfcf",
            facecolor="#f4f4f4",
        )
    _save(fig, out, "convergence")
    plt.close(fig)


def _log_cell_edges(values: list[float]) -> np.ndarray:
    vals = np.array(values, dtype=float)
    if np.any(vals <= 0):
        raise ValueError("log-scaled heatmap values must be positive.")
    if len(vals) == 1:
        factor = np.sqrt(10.0)
        return np.array([vals[0] / factor, vals[0] * factor])

    mid = np.sqrt(vals[:-1] * vals[1:])
    first = vals[0] ** 2 / mid[0]
    last = vals[-1] ** 2 / mid[-1]
    return np.concatenate(([first], mid, [last]))


def plot_heatmap(heatmap_records: list[dict], out: Path) -> None:
    """Test error heatmap: Snacks test error vs (m, lambda)."""
    if not heatmap_records:
        return

    datasets = sorted({r["dataset"] for r in heatmap_records})
    for dataset in datasets:
        ds_records = [r for r in heatmap_records if r["dataset"] == dataset]

        ms = sorted({int(_f(r.get("m"))) for r in ds_records if _f(r.get("m"))})
        lams = sorted(
            {
                lam
                for r in ds_records
                if (lam := _f(r.get("lam"))) is not None and lam <= 5e-3
            }
        )

        if not ms or not lams:
            continue

        # Aggregate by (m, lam): mean test_error across seeds.
        by_config: dict[tuple, list[float]] = defaultdict(list)
        for r in ds_records:
            m = int(_f(r.get("m")))
            lam = _f(r.get("lam"))
            err = _f(r.get("test_error"))
            if m is not None and lam is not None and err is not None:
                by_config[(m, lam)].append(err)

        # Build 2D grid: rows=lam, cols=m.
        data = np.zeros((len(lams), len(ms)))
        for i, lam in enumerate(lams):
            for j, m in enumerate(ms):
                errs = by_config.get((m, lam), [])
                data[i, j] = mean(errs) if errs else np.nan

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanpercentile(finite, 90))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(finite))

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=False)
        im = ax.pcolormesh(
            _log_cell_edges([float(m) for m in ms]),
            _log_cell_edges(lams),
            data,
            cmap="BuRd",
            norm=Normalize(vmin=vmin, vmax=vmax, clip=True),
            shading="auto",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_yticks(lams)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.0e"))
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Number of Nyström columns $m$")
        ax.set_ylabel("Regularization $\\lambda$")
        ax.set_title(f"{dataset} - {PRIMARY_SOLVER_LABEL} test error")
        ax.grid(True, which="major", color="#cfcfcf", linestyle=":", linewidth=0.6)
        ax.grid(True, which="minor", color="#dedede", linestyle=":", linewidth=0.35)
        cbar = plt.colorbar(
            im, ax=ax, extend="max" if float(np.nanmax(finite)) > vmax else "neither"
        )
        cbar.set_label("Test error")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        _save(fig, out, f"heatmap_{dataset.replace('.', '_')}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-csv", default=str(RESULTS_DIR / "diagnostics.csv"))
    parser.add_argument(
        "--heatmap-csv", default=str(RESULTS_DIR / "heatmap_results.csv")
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    diag = _load_csv(Path(args.diag_csv))
    if diag:
        print(f"Loaded {len(diag)} diagnostic rows.")
        plot_convergence(diag, out)
    else:
        print(f"No diagnostics in {args.diag_csv}.")

    heatmap = _load_csv(Path(args.heatmap_csv))
    if heatmap:
        print(f"Loaded {len(heatmap)} heatmap rows.")
        plot_heatmap(heatmap, out)
    else:
        print(f"No heatmap results in {args.heatmap_csv}.")


if __name__ == "__main__":
    main()
