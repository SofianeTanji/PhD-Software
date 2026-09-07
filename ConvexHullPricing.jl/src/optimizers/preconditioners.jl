# ─────────────────────────────────────────────────────────────────────────────
# Preconditioner configurations for PC-BPLM projection metric
# ─────────────────────────────────────────────────────────────────────────────
# The projection objective is: 0.5 * Σ_j w[j] * (π[j] - c[j])²
# Each preconditioner defines how w[j] is computed from gradient history.

abstract type PreconditionerConfig end

# (3) Clipped: clamp((rms[j]+ε)^{-p}, w_min, w_max) with EMA on rms
struct ClippedInversePrecond <: PreconditionerConfig
    beta::Float64
    eps::Float64
    p::Float64
    w_min::Float64
    w_max::Float64
end
ClippedInversePrecond(; p = 2.0, w_min = 1e-4, w_max = 1e4, beta = 0.9, eps = 1e-8) =
    ClippedInversePrecond(beta, eps, p, w_min, w_max)

# ─────────────────────────────────────────────────────────────────────────────
# State initialization
# ─────────────────────────────────────────────────────────────────────────────

init_precond_state(::ClippedInversePrecond, T::Int) = ones(Float64, T)  # rms_grad

# ─────────────────────────────────────────────────────────────────────────────
# Weight computation  (mutates w in-place; may update state)
# ──────────────────────────────────────────────────────────────────────

function compute_weights!(w, cfg::ClippedInversePrecond, state, agg_grad, gradC, K, T)
    rms_grad = state
    @inbounds for t = 1:T
        s2 = cfg.beta * (rms_grad[t]^2) + (1.0 - cfg.beta) * (agg_grad[t]^2)
        rms_grad[t] = sqrt(s2)
        w[t] = clamp((rms_grad[t] + cfg.eps)^(-cfg.p), cfg.w_min, cfg.w_max)
    end
    return w
end