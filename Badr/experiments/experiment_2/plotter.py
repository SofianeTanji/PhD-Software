METHOD_COLORS = {
    "badr": "#C81919",
    "erm": "#662C91",
    "balanced": "#DA9F93",
    "one_fit": "#A0AF63",
    "minmax": "#6699CC",
}
METHOD_MARKERS = {
    "badr": "o",
    "erm": "s",
    "balanced": "^",
    "one_fit": "D",
    "minmax": "x",
}
METHOD_LABELS = {
    "badr": "badr",
    "erm": "Uniform sampling",
    "balanced": "Balanced sampling",
    "one_fit": "One-group fitting",
    "minmax": "Minimax fairness",
}


def plot_results(
    results_path: str = "results_slsqp.jsonl",
    output_path: str = "../../figures/experiment_2_ecdf_qq.pdf",
    alpha: float = 0.8,
    dpi: int = 150,
):
    """Compare test ECDFs with test-set Q-Q plots against badr."""
    import textwrap

    def set_wrapped_ylabel(ax, text, width=10, **label_kwargs):
        wrapped = "\n".join(textwrap.wrap(text, width=width))
        ax.set_ylabel(wrapped, rotation=90, va="center", **label_kwargs)

    from collections import defaultdict
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from tueplots import axes as tp_axes
    from tueplots import figsizes

    def _load_results(path):
        grouped = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                algo = rec.get("algo")
                if algo is not None:
                    grouped[algo].append(rec)
        return grouped

    def _expand_range(rng, frac=0.03):
        """Add relative padding to a (min, max) tuple."""
        if rng is None:
            return None
        lo, hi = rng
        if not np.isfinite(lo) or not np.isfinite(hi):
            return rng
        if hi == lo:
            pad = abs(lo) * frac if lo != 0 else frac
            return lo - pad, hi + pad
        pad = (hi - lo) * frac
        return lo - pad, hi + pad

    res = _load_results(results_path)
    if not res:
        print("No results found.")
        return

    # method_keys = list(res.keys())
    # print(f"Methods found: {method_keys}")
    method_keys = ["badr", "erm", "one_fit", "balanced", "minmax"]
    method_colors = METHOD_COLORS
    method_linestyles = {
        "badr": "solid",
        "erm": "dashed",
        "balanced": "dashdot",
        "one_fit": "solid",
        "minmax": ":",
    }
    method_markers = METHOD_MARKERS
    method_labels = METHOD_LABELS
    method_zorder = {m: (10 if m == "badr" else 1) for m in method_keys}

    metric_names = sorted(
        {
            rec.get("metric")
            for lst in res.values()
            for rec in lst
            if rec.get("metric") is not None
        }
    )

    def _select(method, metric_name):
        return [e for e in res[method] if e.get("metric") == metric_name]

    def _finite(values):
        values = np.asarray(values, dtype=float)
        return values[np.isfinite(values)]

    def _plot_ecdf(ax, values, color, z, linestyle="-"):
        if not values.size:
            return False
        x = np.sort(values)
        y = np.arange(1, x.size + 1) / x.size
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=1.5,
            alpha=alpha,
            linestyle=linestyle,
            zorder=z,
        )
        return True

    def _plot_qq(
        ax,
        reference_values,
        comparison_values,
        color,
        z,
        linestyle,
        marker,
    ):
        """Plot equal-probability empirical quantiles for two methods."""
        if not reference_values.size or not comparison_values.size:
            return False

        n_quantiles = min(reference_values.size, comparison_values.size)
        probabilities = (np.arange(n_quantiles) + 0.5) / n_quantiles
        reference_quantiles = np.quantile(reference_values, probabilities)
        comparison_quantiles = np.quantile(comparison_values, probabilities)
        markevery = max(1, n_quantiles // 18)

        ax.plot(
            reference_quantiles,
            comparison_quantiles,
            color=color,
            linewidth=1.1,
            alpha=alpha,
            linestyle=linestyle,
            marker=marker,
            markersize=2.3,
            markevery=markevery,
            zorder=z,
        )
        return True

    if not metric_names:
        print("No metrics found.")
        return

    with plt.rc_context(figsizes.jmlr2001(nrows=max(1, len(metric_names)), ncols=3)):
        plt.rcParams.update({"figure.dpi": dpi})
        plt.rcParams.update(tp_axes.tick_direction(x="out", y="out"))
        plt.rcParams["font.family"] = "Open Sans"
        plt.rcParams["font.weight"] = "light"
        plt.rcParams["font.size"] = 8.95
        plt.rcParams["axes.facecolor"] = "white"

        fig, axes = plt.subplots(
            nrows=len(metric_names), ncols=2, constrained_layout=True
        )
        axes = [axes] if len(metric_names) == 1 else list(axes)

        legend_handles, legend_labels = [], []

        for i, metric_name in enumerate(metric_names):
            left_ax, right_ax = axes[i]
            test_all, vals = [], {}

            for method in method_keys:
                label = method_labels.get(method, method)
                color = method_colors.get(
                    method, plt.cm.tab10(method_keys.index(method) % 10)
                )
                entries = _select(method, metric_name)
                test_vals = [e["test_metric"] for e in entries if "test_metric" in e]
                test_arr = _finite(test_vals)
                vals[method] = (test_arr, label, color)
                test_all.extend(test_arr.tolist())

            if not test_all:
                left_ax.set_visible(False)
                right_ax.set_visible(False)
                continue

            test_range = _expand_range(
                (min(test_all), max(test_all)) if test_all else None, frac=0.03
            )
            qq_range = test_range
            reference_values = vals["badr"][0]

            for method in method_keys:
                test_vals, label, color = vals[method]
                z = method_zorder.get(method, 1)
                ls = method_linestyles.get(method, "solid")
                marker = method_markers.get(method, "o")

                if i == 0 and test_vals.size:
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linewidth=1.5,
                            alpha=alpha,
                            linestyle=ls,
                            marker=marker,
                            markersize=3,
                        )
                    )
                    legend_labels.append(label)

                _plot_ecdf(left_ax, test_vals, color, z, ls)
                if method != "badr":
                    _plot_qq(
                        right_ax,
                        reference_values,
                        test_vals,
                        color,
                        z,
                        ls,
                        marker,
                    )

            set_wrapped_ylabel(left_ax, metric_name, width=12, labelpad=10)

            if test_range is not None:
                left_ax.set_xlim(test_range)
            left_ax.set_ylim(0, 1)
            left_ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

            if qq_range is not None:
                right_ax.plot(
                    qq_range,
                    qq_range,
                    color="0.25",
                    linewidth=0.8,
                    linestyle=(0, (3, 2)),
                    zorder=0,
                )
                right_ax.set_xlim(qq_range)
                right_ax.set_ylim(qq_range)
            for ax in (left_ax, right_ax):
                ax.grid(
                    True,
                    which="both",
                    axis="both",
                    linestyle=":",
                    linewidth=0.5,
                    alpha=0.6,
                )

        axes[0][0].set_title("Test-set ECDF")
        axes[0][1].set_title("Test-set Q-Q vs. badr")
        axes[len(metric_names) // 2][1].set_ylabel(
            "Comparator quantile", labelpad=7
        )
        axes[-1][0].set_xlabel("Test fairness value")
        axes[-1][1].set_xlabel("badr quantile")

        if legend_handles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.25",
                    linewidth=0.8,
                    linestyle=(0, (3, 2)),
                )
            )
            legend_labels.append("Equal to badr")
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, -0.07),
            )
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_paired_effects(
    results_path: str = "results_slsqp.jsonl",
    output_path: str | None = None,
    split: str = "test",
    tie_threshold: dict | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
    width_mm: float = 115.0,
    height_mm: float = 135.0,
    dpi: int = 150,
):
    """Six-panel paired-effect plot of fairness, baseline minus badr.

    One panel per metric, one row per baseline, one dot per dataset
    (task, state, year, n_groups); positive values mean badr is better.
    Each row shows the median with a 95% cluster-bootstrap interval over
    (task, state) blocks, and badr's win/tie/loss percentages on the right.
    ``split`` selects the train or test values; the axes are shared between
    the two so the figures can be compared. ``tie_threshold`` maps a metric
    name to a practical-equivalence half-width in native units; without it
    ties are exact equalities. The PDF page is exactly ``width_mm`` wide
    (default: the thesis text width, 160 mm minus two 22.5 mm margins).
    Returns the per-row summary statistics.
    """
    from collections import defaultdict
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from tueplots import axes as tp_axes

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if output_path is None:
        output_path = f"../../figures/experiment_2_paired_effects_{split}.pdf"
    baselines = ["erm", "balanced", "one_fit", "minmax"]
    metric_names = [
        "Demographic Parity",
        "Equalized Odds",
        "Equal Opportunity",
        "Disparate Mistreatment",
        "Group Variance",
        "Individual Fairness",
    ]
    tie_threshold = tie_threshold or {}

    # (metric, (task, state, year, n_groups)) -> {algo: {"train": v, "test": v}}
    runs = defaultdict(dict)
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "algo" in rec and "train_metric" in rec and "test_metric" in rec:
                cfg = (rec["task"], rec["state"], rec["year"], rec["n_groups"])
                runs[(rec["metric"], cfg)][rec["algo"]] = {
                    "train": float(rec["train_metric"]),
                    "test": float(rec["test_metric"]),
                }

    def _paired_deltas(metric_name, baseline, which):
        deltas, blocks = [], []
        for (m, cfg), vals in runs.items():
            if m != metric_name or "badr" not in vals or baseline not in vals:
                continue
            d = vals[baseline][which] - vals["badr"][which]
            if np.isfinite(d):
                deltas.append(d)
                blocks.append(f"{cfg[0]}|{cfg[1]}")  # task-state block
        return np.asarray(deltas, dtype=float), np.asarray(blocks)

    rng = np.random.default_rng(seed)

    def _cluster_bootstrap(deltas, blocks, stats):
        """95% percentile intervals of ``stats`` under a block bootstrap."""
        _, inverse = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inverse == i) for i in range(inverse.max() + 1)]
        draws = np.empty((n_bootstrap, len(stats)))
        for k in range(n_bootstrap):
            picks = rng.integers(len(members), size=len(members))
            sample = deltas[np.concatenate([members[p] for p in picks])]
            draws[k] = [stat(sample) for stat in stats]
        return np.quantile(draws, [0.025, 0.975], axis=0).T.tolist()

    summary = []
    stats = {}
    for metric_name in metric_names:
        delta_m = float(tie_threshold.get(metric_name, 0.0))
        for baseline in baselines:
            deltas, blocks = _paired_deltas(metric_name, baseline, split)
            if not deltas.size:
                continue
            wins = deltas > delta_m
            losses = deltas < -delta_m
            ties = ~wins & ~losses
            median_ci, win_rate_ci = _cluster_bootstrap(
                deltas, blocks, [np.median, lambda d, t=delta_m: np.mean(d > t)]
            )
            row = {
                "metric": metric_name,
                "baseline": baseline,
                "split": split,
                "n": int(deltas.size),
                "tie_threshold": delta_m,
                "wins": int(wins.sum()),
                "ties": int(ties.sum()),
                "losses": int(losses.sum()),
                "win_rate": float(wins.mean()),
                "win_rate_ci": win_rate_ci,
                "median": float(np.median(deltas)),
                "median_ci": median_ci,
                "q10": float(np.quantile(deltas, 0.10)),
                "q90": float(np.quantile(deltas, 0.90)),
            }
            summary.append(row)
            stats[(metric_name, baseline)] = (deltas, row)

    if not stats:
        print("No results found.")
        return summary

    row_y = {b: len(baselines) - 1 - i for i, b in enumerate(baselines)}

    def _set_asinh_axis(ax, values, linear_width, min_gap=2.0):
        """asinh x-axis with decade ticks kept ``min_gap`` (asinh units) apart."""
        ax.set_xscale("asinh", linear_width=linear_width)
        lo, hi = (
            np.arcsinh(values.min() / linear_width),
            np.arcsinh(values.max() / linear_width),
        )
        lo, hi = min(lo, -1.0), max(hi, 1.0)
        span = hi - lo
        lo, hi = lo - 0.08 * span, hi + 0.10 * span
        decades = 10.0 ** np.arange(
            np.ceil(np.log10(linear_width)),
            np.floor(np.log10(np.sinh(hi) * linear_width)) + 1,
        )
        ticks, last = [0.0], {1: 0.0, -1: 0.0}
        for d in decades:
            for sign in (1, -1):
                pos = sign * np.arcsinh(d / linear_width)
                if lo < pos < hi and abs(pos - last[sign]) >= min_gap:
                    ticks.append(sign * d)
                    last[sign] = pos
        ticks = sorted(ticks)
        ax.set_xlim(np.sinh(lo) * linear_width, np.sinh(hi) * linear_width)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                "0"
                if t == 0
                else f"${'-' if t < 0 else ''}10^{{{int(np.log10(abs(t)))}}}$"
                for t in ticks
            ]
        )
        ax.xaxis.set_minor_locator(plt.NullLocator())

    nrows, ncols = 3, 2
    with plt.rc_context(
        {
            "figure.figsize": (width_mm / 25.4, height_mm / 25.4),
            "figure.dpi": dpi,
            "font.family": "Open Sans",
            "font.weight": "light",
            "font.size": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "axes.facecolor": "white",
            **tp_axes.tick_direction(x="out", y="out"),
        }
    ):
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, sharey=True, constrained_layout=True
        )
        # h_pad > 0: an "outside" legend otherwise overlaps the x-labels.
        fig.get_layout_engine().set(w_pad=0, h_pad=2 / 72, hspace=0.06, wspace=0.05)
        axes = axes.ravel()

        for ax, metric_name in zip(axes, metric_names):
            rows = [
                (b, stats[(metric_name, b)])
                for b in baselines
                if (metric_name, b) in stats
            ]
            if not rows:
                ax.set_visible(False)
                continue

            # Same axis for the train and test figures.
            axis_values = np.concatenate(
                [
                    _paired_deltas(metric_name, b, which)[0]
                    for b in baselines
                    for which in ("train", "test")
                ]
            )
            nonzero = np.abs(axis_values[axis_values != 0])
            linear_width = float(np.median(nonzero)) if nonzero.size else 1.0
            _set_asinh_axis(ax, axis_values, linear_width)

            delta_m = rows[0][1][1]["tie_threshold"]
            if delta_m > 0:
                ax.axvspan(-delta_m, delta_m, color="0.88", linewidth=0, zorder=0)
            ax.axvline(0, color="0.25", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)

            for baseline, (deltas, row) in rows:
                y = row_y[baseline]
                color = METHOD_COLORS[baseline]
                jitter = rng.uniform(-0.28, 0.28, size=deltas.size)
                ax.scatter(
                    deltas,
                    y + jitter,
                    s=6,
                    color=color,
                    alpha=0.35,
                    linewidths=0,
                    zorder=2,
                )
                lo, hi = row["median_ci"]
                ax.errorbar(
                    row["median"],
                    y,
                    xerr=[[row["median"] - lo], [hi - row["median"]]],
                    fmt={"x": "X"}.get(
                        METHOD_MARKERS[baseline], METHOD_MARKERS[baseline]
                    ),
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    markersize=4.5,
                    elinewidth=1.0,
                    capsize=1.8,
                    zorder=5,
                )

            ax.set_title(metric_name)
            ax.set_yticks([row_y[b] for b in baselines])
            ax.set_yticklabels([METHOD_LABELS[b] for b in baselines])
            ax.set_ylim(-0.6, len(baselines) - 0.4)
            ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
            ax.tick_params(axis="y", length=0)

            # Win / tie / loss percentages as a right-hand axis.
            right = ax.secondary_yaxis("right")
            right.set_yticks([row_y[b] for b, _ in rows])
            right.set_yticklabels(
                [
                    "/".join(
                        f"{100 * row[k] / row['n']:.0f}"
                        for k in ("wins", "ties", "losses")
                    )
                    for _, (_, row) in rows
                ],
                fontsize=7,
            )
            right.tick_params(axis="y", length=0)
            if ax is axes[ncols * (nrows // 2) + ncols - 1]:
                right.set_ylabel("badr wins /\nties / losses (%)", labelpad=6)

        for ax in axes[-ncols:]:
            ax.set_xlabel("baseline − badr")

        fig.legend(
            [
                Line2D(
                    [0], [0], marker="o", color="0.5", linestyle="none", markersize=3
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="0.3",
                    markeredgecolor="black",
                    markersize=4.5,
                    linewidth=1.0,
                ),
                Line2D([0], [0], color="0.25", linewidth=0.8, linestyle=(0, (3, 2))),
            ],
            ["Dataset", "Median, 95% CI", "Equal to badr"],
            loc="outside lower center",
            ncol=3,
        )
        plt.savefig(output_path)
        plt.close(fig)

    return summary


if __name__ == "__main__":
    plot_results(
        results_path="results_slsqp.jsonl",
        output_path="../../figures/experiment_2_ecdf_qq.pdf",
        alpha=1,
    )
    plot_paired_effects(results_path="results_slsqp.jsonl", split="test")
    plot_paired_effects(results_path="results_slsqp.jsonl", split="train")
