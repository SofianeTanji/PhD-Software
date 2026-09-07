# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 8: All table and figure generation
# ═══════════════════════════════════════════════════════════════════════════════

include("common.jl")
using Plots, Printf

const CA_OMIT = Set(["CA9", "CA10", "CA11", "CA12"])

# ── Performance curves (Figure 1) ────────────────────────────────────────────
# Mean gap to L* vs wall time, one plot per dataset.

"""
gap_to_lstar(m, fvals, Ls) → Vector{Float64} (non-negative, same length as fvals)
For dual methods: best-so-far = accumulate(max, fvals), gap = Ls − best.
For CG (primal):  best-so-far = accumulate(min, fvals), gap = best − Ls.
"""
function gap_to_lstar(m::String, fvals::Vector{Float64}, Ls::Float64)
    if m == "CG"
        bsf = accumulate(min, fvals)
        return bsf .- Ls
    else
        bsf = accumulate(max, fvals)
        return Ls .- bsf
    end
end

"""
duality_gap(m, fvals, cg_final) → Vector{Float64} (non-negative, decreasing to 0)
For dual methods: gap = cg_final − best-so-far (dual bound approaches primal from below).
For CG (primal):  gap = best-so-far − cg_final (primal cost approaches minimum from above).
"""
function duality_gap(m::String, fvals::Vector{Float64}, cg_final::Float64)
    if m == "CG"
        bsf = accumulate(min, fvals)
        return bsf .- cg_final
    else
        bsf = accumulate(max, fvals)
        return cg_final .- bsf
    end
end

const PERFORMANCE_TARGET_GAP_FACTOR = 1e-5
const PERFORMANCE_YMAX = Dict("BE" => 1_000.0, "CA" => 160.0)
const PERFORMANCE_YTICKS = Dict(
    "BE" => ([0.0, 200.0, 400.0, 600.0, 800.0, 1_000.0], ["0", "200", "400", "600", "800", "1k"]),
    "CA" => ([0.0, 40.0, 80.0, 120.0, 160.0], ["0", "40", "80", "120", "160"]),
)
const PERFORMANCE_EARLY_CUTOFF = Dict("BE" => 120.0, "CA" => 120.0)
const PERFORMANCE_EARLY_TICKS = Dict(
    "BE" => ([0.0, 0.5, 1.0, 1.5, 2.0], ["0", "0.5", "1", "1.5", "2"]),
    "CA" => ([0.0, 0.5, 1.0, 1.5, 2.0], ["0", "0.5", "1", "1.5", "2"]),
)
const PERFORMANCE_LATE_TICKS = Dict(
    "BE" => ([2.0, 5.0, 10.0, 15.0], ["2", "5", "10", "15"]),
    "CA" => ([2.0, 5.0, 10.0, 15.0], ["2", "5", "10", "15"]),
)
const PERFORMANCE_METHOD_ORDER = [
    "PC-BPLM",
    "PC-BLM",
    "SUBG-L",
    "EST-POL",
    "SUBG",
    "D-Adapt",
    "DoWG",
    "FGM",
    "CG",
]
# Legend is drawn as 3 columns; each column stacks one or more (header, methods)
# groups top-to-bottom. Three wide columns leave room for the long group headers.
const PERFORMANCE_LEGEND_LAYOUT = [
    [("Bundle", ["PC-BPLM", "PC-BLM"]), ("Primal", ["CG"])],
    [("Subgradient", ["EST-POL", "SUBG-L", "SUBG"])],
    [("Smoothing", ["FGM"]), ("Parameter-free", ["D-Adapt", "DoWG"])],
]
const PERFORMANCE_LABELS = Dict(
    "PC-BPLM" => "PC-BPLM",
    "PC-BLM" => "PC-BLM",
    "DLM" => "DLM",
    "SUBG-L" => "SUBG-L",
    "EST-POL" => "SUBG-EP",
    "SUBG" => "SUBG",
    "D-Adapt" => "DA",
    "DoWG" => "DoWG",
    "FGM" => "FGM",
    "CG" => "CG",
)
const PERFORMANCE_STYLES = Dict(
    "PC-BPLM" => (color = "#C97A3D", linetype = :solid, linewidth = 2.8, alpha = 0.98),
    "PC-BLM" => (color = "#E4A874", linetype = :dash, linewidth = 2.7, alpha = 0.98),
    "DLM" => (color = "#8F5D34", linetype = :dashdot, linewidth = 2.7, alpha = 0.98),
    "SUBG-L" => (color = "#315D86", linetype = :dot, linewidth = 2.7, alpha = 0.98),
    "EST-POL" => (color = "#5E82A6", linetype = :solid, linewidth = 2.7, alpha = 0.98),
    "SUBG" => (color = "#9AB4CB", linetype = :dot, linewidth = 2.7, alpha = 0.98),
    "D-Adapt" => (color = "#6F5A8D", linetype = :solid, linewidth = 2.7, alpha = 0.98),
    "DoWG" => (color = "#A88FB9", linetype = :dash, linewidth = 2.7, alpha = 0.98),
    "FGM" => (color = "#5A8F74", linetype = :solid, linewidth = 2.7, alpha = 0.98),
    "CG" => (color = "#9B4F5D", linetype = :solid, linewidth = 2.9, alpha = 0.96),
)

performance_style(method::String) =
    get(PERFORMANCE_STYLES, method, (color = "#111111", linetype = :solid, linewidth = 2.6, alpha = 1.0))
performance_label(method::String) = get(PERFORMANCE_LABELS, method, method)

const PARETO_METHOD_ORDER = [
    "PC-BPLM",
    "PC-BLM",
    "SUBG-L",
    "EST-POL",
    "SUBG",
    "D-Adapt",
    "DoWG",
    "FGM",
    "CG",
]
const PARETO_METHOD_FAMILIES = Dict(
    "PC-BPLM" => :bundle,
    "PC-BLM" => :bundle,
    "SUBG-L" => :subgradient,
    "EST-POL" => :subgradient,
    "SUBG" => :subgradient,
    "D-Adapt" => :parameter_free,
    "DoWG" => :parameter_free,
    "FGM" => :smoothing,
    "CG" => :primal,
)
const PARETO_FAMILY_STYLES = Dict(
    :bundle => (color = "#D08B54", marker = :circle),
    :subgradient => (color = "#4F7AA0", marker = :utriangle),
    :parameter_free => (color = "#816BA0", marker = :diamond),
    :smoothing => (color = "#5A8F74", marker = :hexagon),
    :primal => (color = "#9B4F5D", marker = :rect),
)
const PARETO_METHOD_X_OFFSETS = Dict(
    "PC-BPLM" => -0.05,
    "PC-BLM" => 0.0,
    "EST-POL" => -0.05,
    "SUBG-L" => 0.0,
    "SUBG" => 0.05,
    "D-Adapt" => -0.04,
    "DoWG" => 0.04,
    "FGM" => 0.0,
    "CG" => 0.0,
)
const PARETO_LABEL_Y_OFFSETS = Dict(
    "PC-BPLM" => 0.035,
    "PC-BLM" => -0.015,
    "EST-POL" => 0.040,
    "SUBG-L" => -0.020,
    "SUBG" => 0.050,
    "D-Adapt" => -0.015,
    "DoWG" => 0.020,
    "FGM" => 0.045,
    "CG" => -0.025,
)
const PARETO_Y_TICKS = Dict(
    "BE" => [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1_000.0],
    "CA" => [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 160.0],
)
const PARETO_INSET_XLIMS = Dict(
    "BE" => (0.0, 2.2),
    "CA" => (0.0, 1.4),
)
const PARETO_INSET_YMAX_RAW = Dict(
    "BE" => 1.2,
    "CA" => 0.8,
)
const PARETO_INSET_BOX = (0.48, 0.12, 0.42, 0.36)

pareto_family(method::String) = get(PARETO_METHOD_FAMILIES, method, :other)
pareto_method_x_offset(method::String) = get(PARETO_METHOD_X_OFFSETS, method, 0.0)
pareto_label_y_offset(method::String) = get(PARETO_LABEL_Y_OFFSETS, method, 0.0)

function pareto_family_style(method::String)
    return get(PARETO_FAMILY_STYLES, pareto_family(method), (color = "#111111", marker = :circle))
end

pareto_y_transform(y::Real) = log10(1.0 + max(y, 0.0))

function pareto_tick_label(val::Real)
    if isapprox(val, 1_000.0; atol = 1e-9)
        return "1k"
    elseif isapprox(val, round(val); atol = 1e-9)
        return string(Int(round(val)))
    else
        return @sprintf("%.1f", val)
    end
end

function pareto_y_ticks(ds::String)
    raw_vals = get(PARETO_Y_TICKS, ds, [0.0, 1.0, 10.0, 100.0])
    return pareto_y_transform.(raw_vals), pareto_tick_label.(raw_vals)
end

function draw_performance_category_legend!(
    p;
    header_fontsize = 20,
    label_fontsize = 19,
    y_top = 0.90,
    line_step = 0.175,
    group_gap = 0.035,
)
    ncols = length(PERFORMANCE_LEGEND_LAYOUT)
    col_w = 1 / ncols
    for (col_idx, groups) in enumerate(PERFORMANCE_LEGEND_LAYOUT)
        x_left = (col_idx - 1) * col_w
        hdr_x = x_left + 0.03 * col_w
        seg_x0 = x_left + 0.04 * col_w
        seg_x1 = x_left + 0.16 * col_w
        lab_x = x_left + 0.19 * col_w
        y = y_top
        for (header, methods) in groups
            annotate!(p, hdr_x, y, text(header, header_fontsize, :black, :left))
            y -= line_step
            for method in methods
                style = performance_style(method)
                plot!(
                    p,
                    [seg_x0, seg_x1],
                    [y, y],
                    color = style.color,
                    lw = style.linewidth,
                    ls = style.linetype,
                    alpha = style.alpha,
                    label = "",
                )
                annotate!(p, lab_x, y, text(performance_label(method), label_fontsize, :black, :left))
                y -= line_step
            end
            y -= group_gap
        end
        if col_idx < ncols
            x_div = col_idx * col_w
            plot!(p, [x_div, x_div], [0.08, 0.94], color = :black, alpha = 0.12, lw = 0.8, label = "")
        end
    end
end

function crop_curve_for_plot(curve::Vector{Float64}, ymax::Float64)
    cropped = Vector{Float64}(undef, length(curve))
    for i in eachindex(curve)
        val = curve[i]
        cropped[i] = val # <= ymax ? val : NaN
    end
    return cropped
end

function final_excess_side_payment(method::String, fvals, Ls::Float64)
    isempty(fvals) && return NaN
    final_best = best_so_far(method, fvals)[end]
    excess = method == "CG" ? final_best - Ls : Ls - final_best
    return max(excess, 0.0)
end

function pareto_label_anchor(x::Real, strip_xlo::Real)
    return x >= strip_xlo - 0.10 ? (-0.10, :right) : (0.10, :left)
end

function interquartile_range(vals::Vector{Float64})
    isempty(vals) && return (NaN, NaN)
    sorted = sort(vals)
    return quantile(sorted, 0.25), quantile(sorted, 0.75)
end

function collect_pareto_summaries(
    ds::String,
    tags::Vector{String},
    reference_values::Dict{String,Float64},
    x_boundary::Float64;
    bdir::String = joinpath(RESULTS_DIR, "benchmark"),
)
    summaries = NamedTuple[]
    for method in PARETO_METHOD_ORDER
        style = pareto_family_style(method)
        x_offset = pareto_method_x_offset(method)
        finite_x_raw = Float64[]
        y_raw = Float64[]
        n_censored = 0

        for tag in tags
            haskey(reference_values, tag) || continue
            path = joinpath(bdir, "$(method)_$(tag).jld2")
            isfile(path) || continue
            _, _, fvals, tvec, _ = load_run(path)
            isempty(fvals) && continue

            Ls = reference_values[tag]
            thr = TIME_TO_THRESHOLD_FACTOR * Ls
            final_excess = final_excess_side_payment(method, fvals, Ls)
            isnan(final_excess) && continue

            push!(y_raw, final_excess)
            t_hit = time_to_threshold(tvec, fvals, Ls, thr, method)
            if isfinite(t_hit)
                push!(finite_x_raw, t_hit / 60)
            else
                n_censored += 1
            end
        end

        isempty(y_raw) && continue

        y_center = pareto_y_transform(median(y_raw))
        y_q25, y_q75 = interquartile_range(y_raw)
        y_bar_lo = pareto_y_transform(y_q25)
        y_bar_hi = pareto_y_transform(y_q75)

        has_censoring = n_censored > 0
        if !isempty(finite_x_raw)
            x_center = median(finite_x_raw) + x_offset
            x_q25, x_q75 = interquartile_range(finite_x_raw)
            x_bar_lo = x_q25 + x_offset
            x_bar_hi = x_q75 + x_offset
        else
            x_center = x_boundary + x_offset
            x_bar_lo = NaN
            x_bar_hi = NaN
        end

        push!(summaries, (
            method = method,
            label = performance_label(method),
            color = style.color,
            marker = has_censoring ? :rtriangle : style.marker,
            x_center = x_center,
            y_center = y_center,
            x_bar_lo = x_bar_lo,
            x_bar_hi = x_bar_hi,
            y_bar_lo = y_bar_lo,
            y_bar_hi = y_bar_hi,
            has_censoring = has_censoring,
        ))
    end
    return summaries
end

function summary_in_inset(summary, inset_xlim::Tuple{Float64,Float64}, inset_ymax::Float64)
    return summary.x_center <= inset_xlim[2] && summary.y_center <= inset_ymax
end

function plot_pareto_summary!(
    p,
    summary;
    subplot_idx::Int = 1,
    x_bounds::Tuple{Float64,Float64},
    y_bounds::Tuple{Float64,Float64},
    show_label::Bool,
    label_right_reference::Float64,
    label_fontsize::Int = 8,
)
    if isfinite(summary.x_bar_lo) && isfinite(summary.x_bar_hi)
        plot!(
            p,
            [summary.x_bar_lo, summary.x_bar_hi],
            [summary.y_center, summary.y_center],
            subplot = subplot_idx,
            color = summary.color,
            alpha = 0.80,
            lw = 1.8,
            label = "",
        )
    end
    plot!(
        p,
        [summary.x_center, summary.x_center],
        [summary.y_bar_lo, summary.y_bar_hi],
        subplot = subplot_idx,
        color = summary.color,
        alpha = 0.80,
        lw = 1.8,
        label = "",
    )
    scatter!(
        p,
        [summary.x_center],
        [summary.y_center],
        subplot = subplot_idx,
        color = summary.color,
        markershape = summary.marker,
        markersize = 8.8,
        markerstrokecolor = :white,
        markerstrokewidth = 0.9,
        markeralpha = 0.96,
        label = "",
    )

    show_label || return
    label_dx, halign = pareto_label_anchor(summary.x_center, label_right_reference)
    xlo, xhi = x_bounds
    ylo, yhi = y_bounds
    label_y = clamp(summary.y_center + pareto_label_y_offset(summary.method), ylo + 0.03, yhi - 0.03)
    label_x = clamp(summary.x_center + label_dx, xlo + 0.08, xhi - 0.05)
    annotate!(p, label_x, label_y, text(summary.label, label_fontsize, summary.color, halign), subplot = subplot_idx)
end

function generate_pareto_score_figure(
    ds::String,
    reference_values::Dict{String,Float64};
    bdir::String = joinpath(RESULTS_DIR, "benchmark"),
    figdir::String = joinpath(RESULTS_DIR, "figures"),
)
    tags = ds == "BE" ? [be_tag(i) for i = 1:n_be()] : filter(t -> !(t in CA_OMIT), [ca_tag(i) for i = 1:n_ca()])
    title = ds == "BE" ? "Belgian dataset" : "Californian dataset"

    x_boundary = 15.45
    xlims = (0.0, 15.85)
    xticks = ([0.0, 2.0, 5.0, 10.0, 15.0], ["0", "2", "5", "10", "15"])
    ytick_vals, ytick_labs = pareto_y_ticks(ds)
    y_display_max = maximum(ytick_vals) + 0.10
    inset_xlim = get(PARETO_INSET_XLIMS, ds, (0.0, 2.0))
    inset_ymax_raw = get(PARETO_INSET_YMAX_RAW, ds, 1.0)
    inset_ymax = pareto_y_transform(inset_ymax_raw)
    inset_box_x, inset_box_y, inset_box_w, inset_box_h = PARETO_INSET_BOX
    summaries = collect_pareto_summaries(ds, tags, reference_values, x_boundary; bdir = bdir)

    p = plot(
        title = title,
        xlabel = "Time to threshold (min)",
        ylabel = "Final excess side payments at 15 min (\$)",
        xticks = xticks,
        yticks = (ytick_vals, ytick_labs),
        xlims = xlims,
        ylims = (0.0, y_display_max),
        legend = false,
        dpi = 200,
        size = (760, 520),
        left_margin = 15Plots.mm,
        right_margin = 5Plots.mm,
        bottom_margin = 7Plots.mm,
        top_margin = 5Plots.mm,
        gridalpha = 0.18,
        guidefontsize = 13,
        tickfontsize = 10,
        titlefontsize = 11,
        widen = false,
    )

    vline!(p, [15.0], color = :black, alpha = 0.16, lw = 1.2, ls = :dash, label = "")
    annotate!(p, x_boundary, y_display_max - 0.04, text(">900 s", 8, :black, :center))
    annotate!(
        p,
        0.98 * xlims[2],
        0.08,
        text("right-pointing markers indicate censored methods", 8, :black, :right),
    )

    for summary in summaries
        plot_pareto_summary!(
            p,
            summary;
            subplot_idx = 1,
            x_bounds = xlims,
            y_bounds = (0.0, y_display_max),
            show_label = !summary_in_inset(summary, inset_xlim, inset_ymax),
            label_right_reference = x_boundary,
            label_fontsize = 8,
        )
    end

    plot!(
        p,
        [inset_xlim[1], inset_xlim[2], inset_xlim[2], inset_xlim[1], inset_xlim[1]],
        [0.0, 0.0, inset_ymax, inset_ymax, 0.0],
        color = :black,
        alpha = 0.30,
        lw = 1.0,
        ls = :dash,
        label = "",
    )

    inset_xticks = ds == "BE" ? ([0.0, 1.0, 2.0], ["0", "1", "2"]) : ([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    inset_ytick_raw = ds == "BE" ? [0.0, 0.3, 1.0] : [0.0, 0.2, 0.5]
    inset_yticks = (pareto_y_transform.(inset_ytick_raw), pareto_tick_label.(inset_ytick_raw))

    plot!(
        p,
        inset = (1, bbox(inset_box_x, inset_box_y, inset_box_w, inset_box_h)),
        subplot = 2,
        xlims = inset_xlim,
        ylims = (0.0, inset_ymax),
        xticks = inset_xticks,
        yticks = inset_yticks,
        bg_inside = RGBA(1, 1, 1, 0.95),
        framestyle = :box,
        gridalpha = 0.14,
        guidefontsize = 8,
        tickfontsize = 7,
        titlefontsize = 8,
        widen = false,
        title = "Zoom",
        xlabel = "",
        ylabel = "",
        legend = false,
    )
    for summary in summaries
        summary_in_inset(summary, inset_xlim, inset_ymax) || continue
        plot_pareto_summary!(
            p,
            summary;
            subplot_idx = 2,
            x_bounds = inset_xlim,
            y_bounds = (0.0, inset_ymax),
            show_label = true,
            label_right_reference = inset_xlim[2] - 0.10,
            label_fontsize = 7,
        )
    end

    outpath = joinpath(figdir, "pareto_score_$ds.pdf")
    savefig(p, outpath)
    @info "Saved $outpath"
    return p
end

function generate_pareto_score_figures()
    baseline_lstar = load_ground_truth()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    figdir = joinpath(RESULTS_DIR, "figures")
    mkpath(figdir)

    tags_by_ds = Dict(
        "BE" => [be_tag(i) for i = 1:n_be()],
        "CA" => filter(t -> !(t in CA_OMIT), [ca_tag(i) for i = 1:n_ca()]),
    )
    analysis_tags = vcat(tags_by_ds["BE"], tags_by_ds["CA"])
    reference_values = Dict(
        tag => promoted_reference_value(tag, baseline_lstar[tag]; bdir = bdir) for
        tag in analysis_tags if haskey(baseline_lstar, tag)
    )
    return Dict(ds => generate_pareto_score_figure(ds, reference_values; bdir = bdir, figdir = figdir) for ds in ["BE", "CA"])
end

# Per-dataset mean duality-gap curves on a common time grid, plus the plotting
# metadata (ticks, limits, target threshold) needed to render the panels.
function performance_dataset_curves(ds::String, ntag::Int, mktagf, bdir::String)
    tags = filter(t -> !(t in CA_OMIT), [mktagf(i) for i = 1:ntag])
    t_grid = range(0.0, BUDGET, length = 300)
    x_plot_min = collect(t_grid) ./ 60

    cg_finals = Dict{String,Float64}()
    for tag in tags
        cg_path = joinpath(bdir, "CG_$tag.jld2")
        isfile(cg_path) || continue
        _, _, cg_fvals, _, _ = load_run(cg_path)
        isempty(cg_fvals) && continue
        cg_finals[tag] = minimum(cg_fvals)
    end

    mean_cg = isempty(cg_finals) ? NaN : mean(values(cg_finals))
    mean_curves = Dict{String,Vector{Float64}}()
    for m in PERFORMANCE_METHOD_ORDER
        curves = Vector{Float64}[]
        for tag in tags
            path = joinpath(bdir, "$(m)_$(tag).jld2")
            isfile(path) || continue
            haskey(cg_finals, tag) || continue
            _, _, fvals, tvec, _ = load_run(path)
            isempty(fvals) && continue

            gap = duality_gap(m, fvals, cg_finals[tag])
            gap_on_grid = interpolate_onto_grid(tvec[2:end], gap, collect(t_grid))
            push!(curves, gap_on_grid)
        end
        isempty(curves) && continue
        mean_curves[m] = mean(curves)
    end

    ymax = get(PERFORMANCE_YMAX, ds, 1_000.0)
    ytick_vals, ytick_labs = get(
        PERFORMANCE_YTICKS,
        ds,
        ([0.0, ymax / 2, ymax], ["0", @sprintf("%.0f", ymax / 2), @sprintf("%.0f", ymax)]),
    )
    ranked_methods = sort(
        collect(keys(mean_curves));
        by = m -> (mean_curves[m][end], performance_label(m)),
    )
    threshold = isnan(mean_cg) ? NaN : PERFORMANCE_TARGET_GAP_FACTOR * mean_cg
    early_cutoff_min = get(PERFORMANCE_EARLY_CUTOFF, ds, 60.0) / 60
    early_xticks = get(
        PERFORMANCE_EARLY_TICKS,
        ds,
        ([0.0, early_cutoff_min / 2, early_cutoff_min], ["0", @sprintf("%.1f", early_cutoff_min / 2), @sprintf("%.1f", early_cutoff_min)]),
    )
    late_xticks = get(
        PERFORMANCE_LATE_TICKS,
        ds,
        ([early_cutoff_min, 5.0, 10.0, 15.0], [@sprintf("%.1f", early_cutoff_min), "5", "10", "15"]),
    )

    return (
        x = x_plot_min,
        mean_curves = mean_curves,
        ranked_methods = ranked_methods,
        ymax = ymax,
        ytick_vals = ytick_vals,
        ytick_labs = ytick_labs,
        threshold = threshold,
        early_cutoff_min = early_cutoff_min,
        early_xticks = early_xticks,
        late_xticks = late_xticks,
    )
end

# Full-time-range convergence panel for one dataset, on a log-scaled wall-time
# x-axis (the early and later regimes are merged into a single panel).
const PERFORMANCE_LOG_XLIMS = (0.045, 15.5)
const PERFORMANCE_LOG_XTICKS = ([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0], ["0.1", "0.5", "1", "2", "5", "10", "15"])

function performance_panel(d; title::String, ylabel::String, show_xlabel::Bool, margins, fonts)
    # Drop the t = 0 grid point, which has no place on a log axis.
    pos = d.x .> 0

    p = plot(
        title = title,
        xlabel = show_xlabel ? "Time (min)" : "",
        ylabel = ylabel,
        xscale = :log10,
        yticks = (d.ytick_vals, d.ytick_labs),
        xticks = PERFORMANCE_LOG_XTICKS,
        xlims = PERFORMANCE_LOG_XLIMS,
        ylims = (0.0, d.ymax),
        legend = false,
        dpi = 200,
        left_margin = margins.left,
        right_margin = margins.right,
        bottom_margin = margins.bottom,
        top_margin = margins.top,
        gridalpha = 0.18,
        guidefontsize = fonts.guide,
        tickfontsize = fonts.tick,
        titlefontsize = fonts.title,
        widen = false,
    )

    for m in d.ranked_methods
        style = performance_style(m)
        plot!(
            p,
            d.x[pos],
            crop_curve_for_plot(d.mean_curves[m], d.ymax)[pos],
            label = "",
            lw = style.linewidth,
            color = style.color,
            ls = style.linetype,
            alpha = style.alpha,
        )
    end
    if !isnan(d.threshold)
        hline!(p, [d.threshold], color = :black, lw = 2.4, ls = :dashdot, label = "")
    end
    return p
end

function generate_performance_curves()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    mkpath(joinpath(RESULTS_DIR, "figures"))

    # Font sizes are tuned so that, when the figure is placed at the 115 mm text
    # width of the thesis, axis text reads ≈ 7 pt. With a 1000 pt canvas the scale
    # factor is 326/1000 ≈ 0.33, so native sizes are ≈ 3× the on-page size.
    fonts = (guide = 21, tick = 20, title = 24)

    datasets = [
        (ds = "BE", ntag = n_be(), mktagf = be_tag, currency = "€", name = "Belgian instances"),
        (ds = "CA", ntag = n_ca(), mktagf = ca_tag, currency = "\$", name = "Californian instances"),
    ]

    panels = Any[]
    for (idx, info) in enumerate(datasets)
        d = performance_dataset_curves(info.ds, info.ntag, info.mktagf, bdir)
        is_bottom = idx == length(datasets)
        push!(panels, performance_panel(
            d;
            title = info.name,
            ylabel = "Excess side payments ($(info.currency))",
            show_xlabel = is_bottom,
            margins = (left = 16Plots.mm, right = 5Plots.mm, bottom = is_bottom ? 9Plots.mm : 5Plots.mm, top = 6Plots.mm),
            fonts = fonts,
        ))
    end

    p_leg = plot(
        framestyle = :none,
        grid = false,
        showaxis = false,
        ticks = nothing,
        legend = false,
        xlims = (0.0, 1.0),
        ylims = (0.0, 1.0),
        left_margin = 2Plots.mm,
        right_margin = 2Plots.mm,
        top_margin = 2Plots.mm,
        bottom_margin = 2Plots.mm,
    )
    draw_performance_category_legend!(p_leg)

    p = plot(
        panels[1], panels[2], p_leg;
        layout = @layout([a; b; c{0.22h}]),
        size = (1_000, 1_280),
    )

    outpath = joinpath(RESULTS_DIR, "figures", "performance_curves.pdf")
    savefig(p, outpath)
    @info "Saved performance_curves.pdf"
end


function generate_performance_curves_all()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    mkpath(joinpath(RESULTS_DIR, "figures"))

    ytick_vals = [0.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
    ytick_labs = ["0", "10", "100", "1k", "10k", "100k"]
    xtick_mins = [0, 1, 2, 5, 10, 15]
    xtick_labs = string.(xtick_mins)

    for (ds, ntag, mktagf, nrows, ncols, ds_title) in [
        ("BE", n_be(), be_tag, 2, 4, "Belgian"),
        ("CA", n_ca(), ca_tag, 4, 4, "Californian"),
    ]
        tags = filter(t -> !(t in CA_OMIT), [mktagf(i) for i = 1:ntag])
        t_grid = range(1.0, BUDGET, length = 300)
        x_plot = collect(t_grid) ./ 60

        cg_finals = Dict{String,Float64}()
        for tag in tags
            cg_path = joinpath(bdir, "CG_$tag.jld2")
            isfile(cg_path) || continue
            _, _, cg_fvals, _, _ = load_run(cg_path)
            isempty(cg_fvals) && continue
            cg_finals[tag] = minimum(cg_fvals)
        end

        subplots = Any[]
        for (idx, tag) in enumerate(tags)
            row, col = divrem(idx - 1, ncols)
            sp_title =
                haskey(cg_finals, tag) ? "$tag ($(@sprintf("%.2e", cg_finals[tag])) \$)" :
                tag
            sp = plot(
                title = sp_title,
                xlabel = row == nrows - 1 ? "Time (min)" : "",
                ylabel = col == 0 ? "Excess side payments (\$)" : "",
                yticks = (
                    ytick_vals,
                    col == 0 ? ytick_labs : fill("", length(ytick_vals)),
                ),
                xticks = (
                    float.(xtick_mins),
                    row == nrows - 1 ? xtick_labs : fill("", length(xtick_mins)),
                ),
                xlims = (0.0, 15.0),
                legend = idx == 1 ? :topright : false,
            )
            haskey(cg_finals, tag) || (push!(subplots, sp); continue)
            for m in METHOD_NAMES
                path = joinpath(bdir, "$(m)_$(tag).jld2")
                isfile(path) || continue
                _, _, fvals, tvec, _ = load_run(path)
                isempty(fvals) && continue
                gap = duality_gap(m, fvals, cg_finals[tag])
                gap_on_grid = interpolate_onto_grid(tvec[2:end], gap, collect(t_grid))
                plot!(sp, x_plot, gap_on_grid, label = m, lw = 1.5)
            end
            hline!(
                sp,
                [1e-5 * cg_finals[tag]],
                color = :black,
                lw = 1.5,
                ls = :solid,
                label = "",
            )
            push!(subplots, sp)
        end

        p = plot(
            subplots...,
            layout = (nrows, ncols),
            plot_title = ds_title,
            size = (ncols * 380, nrows * 280),
            left_margin = 10Plots.mm,
            bottom_margin = 6Plots.mm,
            dpi = 200,
        )
        savefig(p, joinpath(RESULTS_DIR, "figures", "performance_curves_all_$ds.pdf"))
        @info "Saved performance_curves_all_$ds.pdf"
    end
end


# ── Accuracy tables (Tables 4 & 5) ──────────────────────────────────────────
# Per-instance excess side payments after 15 min and 5 min.

function generate_accuracy_tables()
    baseline_lstar = load_ground_truth()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    tags = all_tags()
    reference_values = Dict(
        tag => promoted_reference_value(tag, baseline_lstar[tag]; bdir = bdir) for
        tag in tags if haskey(baseline_lstar, tag)
    )

    for (cutoff, label) in [(BUDGET, "15min"), (BUDGET_5, "5min")]
        df = DataFrame(instance = String[])
        for m in METHOD_NAMES
            df[!, m] = Float64[]
        end

        for tag in tags
            haskey(reference_values, tag) || continue
            Ls = reference_values[tag]
            row = Any[tag]
            for m in METHOD_NAMES
                path = joinpath(bdir, "$(m)_$(tag).jld2")
                if !isfile(path)
                    push!(row, NaN)
                    continue
                end
                _, _, fvals, tvec, _ = load_run(path)
                bsf_at_cutoff = best_so_far_at_cutoff(m, fvals, tvec, cutoff)
                # For CG: excess = bsf - Ls (how much worse than optimal)
                # For dual: excess = Ls - bsf (duality gap)
                excess = m == "CG" ? bsf_at_cutoff - Ls : Ls - bsf_at_cutoff
                push!(row, excess)
            end
            push!(df, row)
        end

        jldsave(joinpath(RESULTS_DIR, "accuracy_table_$label.jld2"); df, reference_values)
        @info "Accuracy table ($label, excess side payments): $(nrow(df)) instances × $(length(METHOD_NAMES)) methods"
    end
end


# ── Time-to-threshold (Table 6) ─────────────────────────────────────────────
# Time to reach an excess side payment threshold equal to 1e-5 × L*.

function generate_time_to_threshold()
    baseline_lstar = load_ground_truth()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    tags = all_tags()
    reference_values = Dict(
        tag => promoted_reference_value(tag, baseline_lstar[tag]; bdir = bdir) for
        tag in tags if haskey(baseline_lstar, tag)
    )
    threshold_values = Dict(
        tag => TIME_TO_THRESHOLD_FACTOR * reference_values[tag] for
        tag in keys(reference_values)
    )

    df = DataFrame(instance = String[])
    for m in METHOD_NAMES
        df[!, m] = Float64[]
    end

    for tag in tags
        haskey(reference_values, tag) || continue
        Ls = reference_values[tag]
        thr = threshold_values[tag]
        row = Any[tag]
        for m in METHOD_NAMES
            path = joinpath(bdir, "$(m)_$(tag).jld2")
            if !isfile(path)
                push!(row, Inf)
                continue
            end
            _, _, fvals, tvec, _ = load_run(path)
            push!(row, time_to_threshold(tvec, fvals, Ls, thr, m))
        end
        push!(df, row)
    end

    # Geometric means (excluding Inf)
    geo = Dict{String,Float64}()
    for m in METHOD_NAMES
        finite = filter(isfinite, df[!, m])
        geo[m] = isempty(finite) ? Inf : exp(mean(log.(finite)))
    end

    jldsave(joinpath(RESULTS_DIR, "time_to_threshold.jld2"); df, geo, reference_values, threshold_values)
    @info "Time-to-threshold geometric means:"
    for (k, v) in sort(collect(geo); by = last)
        @info "  $k: $(round(v; digits=1))s"
    end
    return df, geo
end


# ── Oracle time split (R6, R7) ──────────────────────────────────────────────
# From instrumented benchmark results: oracle vs master time breakdown.

function generate_oracle_time_split()
    bdir = joinpath(RESULTS_DIR, "benchmark")
    dir = joinpath(RESULTS_DIR, "oracle_time_split")
    mkpath(dir)
    tags = all_tags()

    results = DataFrame(
        method = String[],
        tag = String[],
        total_time = Float64[],
        total_oracle = Float64[],
        total_master = Float64[],
        oracle_frac = Float64[],
        n_oracle_calls = Int[],
    )

    for m in METHOD_NAMES, tag in tags
        path = joinpath(bdir, "$(m)_$(tag).jld2")
        isfile(path) || continue
        _, _, fvals, tvec, oracle_times = load_run(path)
        isempty(oracle_times) && continue

        total_time = tvec[end]
        total_oracle = sum(oracle_times)
        total_master = total_time - total_oracle
        oracle_frac = total_time > 0 ? total_oracle / total_time : 0.0
        n_calls = length(oracle_times)
        push!(
            results,
            (m, tag, total_time, total_oracle, total_master, oracle_frac, n_calls),
        )
    end

    jldsave(joinpath(dir, "time_split.jld2"); results)
    @info "Oracle time split: $(nrow(results)) entries."
    return results
end


# ── MILP statistics table (R5) ──────────────────────────────────────────────

function generate_milp_statistics_table()
    profiling_path = joinpath(RESULTS_DIR, "milp_profiling", "milp_profiling.jld2")
    if !isfile(profiling_path)
        @warn "MILP profiling results not found. Run run_milp_profiling() first."
        return nothing
    end
    prof = load(profiling_path)["results"]

    results = DataFrame(
        dataset = String[],
        mean_vars = Float64[],
        mean_constrs = Float64[],
        mean_solve = Float64[],
        std_solve = Float64[],
        median_solve = Float64[],
        max_solve = Float64[],
    )

    for ds in unique(prof.dataset)
        sub = filter(r -> r.dataset == ds, prof)
        push!(
            results,
            (
                ds,
                mean(sub.n_vars),
                mean(sub.n_constrs),
                mean(sub.solve_time),
                std(sub.solve_time),
                median(sub.solve_time),
                maximum(sub.solve_time),
            ),
        )
    end

    jldsave(joinpath(RESULTS_DIR, "milp_statistics.jld2"); results)
    @info "MILP statistics table generated."
    return results
end


# ── FGM smoothing sensitivity table (R2) ────────────────────────────────────

function generate_fgm_sensitivity_table()
    sens_path = joinpath(RESULTS_DIR, "fgm_smoothing", "sensitivity.jld2")
    if !isfile(sens_path)
        @warn "FGM sensitivity results not found. Run run_fgm_smoothing_sensitivity() first."
        return nothing
    end
    sens = load(sens_path)["results"]
    Lstar = load_ground_truth()

    results = DataFrame(
        dataset = String[],
        smoothing = Float64[],
        mean_gap = Float64[],
        std_gap = Float64[],
        n_instances = Int[],
    )

    for ds in unique(sens.dataset)
        sub_ds = filter(r -> r.dataset == ds, sens)
        for ς in sort(unique(sub_ds.smoothing))
            sub_s = filter(r -> r.smoothing == ς, sub_ds)
            gaps = Float64[]
            for tag in unique(sub_s.tag)
                haskey(Lstar, tag) || continue
                Ls = Lstar[tag]
                # best objective over all η choices for this (tag, ς)
                sub_t = filter(r -> r.tag == tag, sub_s)
                best = maximum(sub_t.best_obj)
                push!(gaps, (Ls - best) / abs(Ls))
            end
            isempty(gaps) && continue
            push!(results, (ds, ς, mean(gaps), std(gaps), length(gaps)))
        end
    end

    jldsave(joinpath(RESULTS_DIR, "fgm_sensitivity_table.jld2"); results)
    @info "FGM sensitivity table generated: $(nrow(results)) rows."
    return results
end


# ── Time split table (R6/R7) ────────────────────────────────────────────────

function generate_time_split_table()
    split_path = joinpath(RESULTS_DIR, "oracle_time_split", "time_split.jld2")
    if !isfile(split_path)
        @info "Running oracle time split analysis..."
        generate_oracle_time_split()
    end
    split_data = load(split_path)["results"]

    results = DataFrame(
        method = String[],
        dataset = String[],
        mean_oracle_frac = Float64[],
        mean_master_frac = Float64[],
        mean_calls = Float64[],
    )

    for m in METHOD_NAMES
        for ds in ["BE", "CA"]
            sub = filter(r -> r.method == m && startswith(r.tag, ds), split_data)
            nrow(sub) == 0 && continue
            push!(
                results,
                (
                    m,
                    ds,
                    mean(sub.oracle_frac),
                    1.0 - mean(sub.oracle_frac),
                    mean(sub.n_oracle_calls),
                ),
            )
        end
    end

    jldsave(joinpath(RESULTS_DIR, "time_split_table.jld2"); results)
    @info "Time split table: $(nrow(results)) entries."
    return results
end


# ── Parameter robustness table (R8) ─────────────────────────────────────────

function generate_parameter_robustness_table()
    rob_path = joinpath(RESULTS_DIR, "robustness", "robustness.jld2")
    if !isfile(rob_path)
        @warn "Robustness results not found. Run run_parameter_robustness() first."
        return nothing
    end
    rob = load(rob_path)["results"]

    results = DataFrame(
        method = String[],
        dataset = String[],
        best_param = Float64[],
        best_obj = Float64[],
        worst_param = Float64[],
        worst_obj = Float64[],
        error_ratio = Float64[],
    )

    for m in unique(rob.method)
        for ds in ["BE", "CA"]
            sub = filter(r -> r.method == m && r.dataset == ds, rob)
            nrow(sub) == 0 && continue
            # Group by param, take mean obj across instances
            param_means = Dict{Float64,Float64}()
            for α in unique(sub.param)
                param_means[α] = mean(filter(r -> r.param == α, sub).best_obj)
            end
            best_α = argmax(param_means)
            worst_α = argmin(param_means)
            push!(
                results,
                (
                    m,
                    ds,
                    best_α,
                    param_means[best_α],
                    worst_α,
                    param_means[worst_α],
                    abs(param_means[best_α]) > 0 ?
                    abs(param_means[worst_α] / param_means[best_α]) : NaN,
                ),
            )
        end
    end

    jldsave(joinpath(RESULTS_DIR, "robustness_table.jld2"); results)
    @info "Parameter robustness table: $(nrow(results)) entries."
    return results
end


# ── Bundle stopping table (R9) ──────────────────────────────────────────────

function generate_bundle_convergence_table()
    gap_path = joinpath(RESULTS_DIR, "bundle_gaps", "gap_analysis.jld2")
    if !isfile(gap_path)
        @warn "Bundle gap results not found. Run run_bundle_gap_analysis() first."
        return nothing
    end
    gaps = load(gap_path)["results"]

    results = DataFrame(
        method = String[],
        mean_gap = Float64[],
        median_gap = Float64[],
        frac_target_reached = Float64[],
        mean_time_reached = Float64[],
    )

    for m in ["PC-BLM", "PC-BPLM"]
        sub = filter(r -> r.method == m, gaps)
        nrow(sub) == 0 && continue
        reached_times = filter(isfinite, sub.time_to_target)
        push!(
            results,
            (
                m,
                mean(sub.final_gap),
                median(sub.final_gap),
                mean(sub.reached_target),
                isempty(reached_times) ? NaN : mean(reached_times),
            ),
        )
    end

    jldsave(joinpath(RESULTS_DIR, "bundle_convergence_table.jld2"); results)
    @info "Bundle convergence table: $(nrow(results)) methods."
    return results
end


# ── Preconditioner comparison figure ────────────────────────────────────────

function generate_precond_comparison_figure()
    dir = joinpath(RESULTS_DIR, "precond_comparison")
    figdir = joinpath(RESULTS_DIR, "figures")
    mkpath(figdir)

    budget = 90.0  # seconds
    t_grid = range(0.5, budget, length = 300)
    x_plot = collect(t_grid) ./ 60

    ytick_vals = [0.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
    ytick_labs = ["0", "10", "100", "1k", "10k", "100k"]
    xtick_secs = [0, 15, 30, 45, 60, 90]
    xtick_labs = ["0", "15s", "30s", "45s", "1m", "1.5m"]

    Lstar = load_ground_truth()

    # Discover which methods have results on disk
    all_files = readdir(dir)
    methods = sort(
        unique([
            replace(f, r"_[A-Z]+[0-9]+\.jld2$" => "") for
            f in all_files if endswith(f, ".jld2") && f != "precond_comparison.jld2"
        ]),
    )

    instances = [
        ("BE1", "BE"),
        ("BE3", "BE"),
        ("BE5", "BE"),
        ("CA1", "CA"),
        ("CA5", "CA"),
        ("CA13", "CA"),
    ]

    nrows, ncols = 2, 3
    subplots = Any[]
    for (idx, (tag, _)) in enumerate(instances)
        row, col = divrem(idx - 1, ncols)
        lstar_val = get(Lstar, tag, nothing)
        sp_title = isnothing(lstar_val) ? tag : "$tag (L*=$(@sprintf("%.2e", lstar_val)))"
        sp = plot(
            title = sp_title,
            xlabel = row == nrows - 1 ? "Wall time" : "",
            ylabel = col == 0 ? "Excess side payments (\$)" : "",
            yticks = (
                ytick_vals,
                col == 0 ? ytick_labs : fill("", length(ytick_vals)),
            ),
            xticks = (
                float.(xtick_secs) ./ 60,
                row == nrows - 1 ? xtick_labs : fill("", length(xtick_secs)),
            ),
            xlims = (0.0, budget / 60),
            legend = idx == 1 ? :topright : false,
        )

        isnothing(lstar_val) && (push!(subplots, sp); continue)

        for m in methods
            path = joinpath(dir, "$(m)_$(tag).jld2")
            isfile(path) || continue
            _, _, fvals, tvec, _ = load_run(path)
            isempty(fvals) && continue
            gap = gap_to_lstar(m == "MC-BPLM" ? "other" : m, fvals, lstar_val)
            gap_on_grid = interpolate_onto_grid(tvec[2:end], gap, collect(t_grid))
            plot!(sp, x_plot, gap_on_grid, label = m, lw = 1.5)
        end
        push!(subplots, sp)
    end

    p = plot(
        subplots...,
        layout = (nrows, ncols),
        plot_title = "Preconditioner comparison (90s budget)",
        size = (ncols * 380, nrows * 280),
        left_margin = 10Plots.mm,
        bottom_margin = 6Plots.mm,
        dpi = 200,
    )
    outpath = joinpath(figdir, "precond_comparison_all.pdf")
    savefig(p, outpath)
    @info "Saved $outpath"
    return p
end

# ── Master convenience ──────────────────────────────────────────────────────

function generate_all_tables_and_figures()
    generate_accuracy_tables()
    generate_time_to_threshold()
    generate_performance_curves()
    generate_pareto_score_figures()
    generate_performance_curves_all()
    generate_oracle_time_split()
    generate_time_split_table()
    generate_milp_statistics_table()
    generate_fgm_sensitivity_table()
    generate_parameter_robustness_table()
    generate_bundle_convergence_table()
end
