function davis_yin_gradient_mapping_bound(; k, initial, L, γ, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    γf = _require_finite_positive(γ, :γ)
    known_gamma = _known_real(γf)
    known_L = _known_real(Lf)
    known_gamma === nothing ||
        known_L === nothing ||
        known_gamma < 2 / known_L ||
        throw(DomainError(γ, "requires γ < 2 / L"))
    return 2 * R2 / (γf^2 * (2 - γf * Lf) * (k + 1))
end

davis_yin_stepsize(; L, kwargs...) = 4 / (3 * _require_finite_positive(L, :L))

function chambolle_pock_balanced_step(; operator_norm, kwargs...)
    return 1 / _require_finite_positive(operator_norm, :operator_norm)
end

function balanced_primal_dual_primal_step(; initial, L, M, kwargs...)
    radius = sqrt(_require_finite_positive(initial, :initial))
    smoothness = _require_finite_positive(L, :L)
    coupling = _require_finite_positive(M, :M)
    return 1 / (smoothness + coupling / radius)
end

function balanced_primal_dual_dual_step(; initial, M, operator_norm, kwargs...)
    radius = sqrt(_require_finite_positive(initial, :initial))
    coupling = _require_finite_positive(M, :M)
    norm = _require_finite_positive(operator_norm, :operator_norm)
    return coupling / (norm^2 * radius)
end

# Chambolle-Pock ergodic primal-dual gap for min_x f(x) + g(Mx), f and g convex
# proximable, ‖M‖ = operator_norm. With the balanced optimal steps τ = σ = 1/‖M‖
# (so τσ‖M‖² = 1), Bousselmi-Hendrickx-Glineur 2023 (arXiv:2302.08781, Thm 4.8)
# give the tight ergodic gap L(x̄_N,u) − L(x,ū_N) ≤ ‖M‖ · R² / (2(N+1)), where the
# primal-dual initial distance R² is carried in `initial`.
function chambolle_pock_ergodic_gap_bound(; k, initial, operator_norm, kwargs...)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    LM = _require_finite_positive(operator_norm, :operator_norm)
    return LM * R2 / (2 * (k + 1))
end

# Chambolle-Pock with a FORWARD (gradient) step on the smooth term, for
#     min_x f(x) + g(Mx),   f convex and L_f-smooth,   g convex and B_g-Lipschitz.
# Steps σ = ρ/‖M‖, τ = 1/(L_f + ρ‖M‖) satisfy τL_f + τσ‖M‖² ≤ 1, giving the exact
# ergodic saddle bound  L(X_N,y) − L(x,Y_N) ≤ [(L_f+ρ‖M‖)R_x² + (‖M‖/ρ)R_y²]/(2N).
# Balancing at ρ* = R_y/R_x gives  [L_f R_x²/2 + ‖M‖R_x R_y]/N. A saddle bound is
# only a primal bound when dom g* is bounded: g being B_g-Lipschitz gives
# dom g* ⊆ {‖y‖ ≤ B_g}, so with y⁰ = 0 and R_y = B_g,
#     F(X_N) − F(x*) ≤ [L_f R_x²/2 + ‖M‖ B_g R_x] / N.
# Argo's composition calculus already reports lipschitz(g∘M) = B_g‖M‖, which is
# exactly the primal-dual coupling constant, so no separate B_g is needed.
function chambolle_pock_forward_ergodic_gap_bound(; k, initial, L, M, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    coupling = _require_finite_positive(M, :M)      # B_g‖M‖
    return (Lf * R2 / 2 + coupling * sqrt(R2)) / k
end

# Condat-Vũ ergodic primal bound for  min_x f(x) + g(Mx),  f convex and L_f-smooth,
# g convex and G-Lipschitz (so dom g* ⊆ {‖y‖ ≤ G}). The iteration takes the primal
# step first, using the previous dual iterate:
#     x⁺ = x − τ(∇f(x) + Mᵀy),   y⁺ = prox_{σg*}(y + σM(2x⁺ − x)),
# under the finite-time step condition 1/τ ≥ L_f + σ‖M‖². Saturating it and
# optimising, σ* = G/(‖M‖R), τ* = 1/(L_f + ‖M‖G/R), gives
#     F(X_N) − F(x*) ≤ [L_f R²/2 + ‖M‖GR + G‖M(x* − x⁰)‖] / N
#                    ≤ [L_f R²/2 + 2‖M‖GR] / N        (using ‖M(x*−x⁰)‖ ≤ ‖M‖R).
# The extra ‖M‖GR relative to the forward-step Chambolle-Pock bound is the price of
# the initial primal-dual cross term; it vanishes when M(x* − x⁰) = 0. As with CP,
# ‖M‖G is exactly the composed Lipschitz constant of g∘M that Argo already derives.
function condat_vu_ergodic_gap_bound(; k, initial, L, M, kwargs...)
    _require_integer_at_least(k, 1)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    coupling = _require_finite_positive(M, :M)      # G‖M‖
    return (Lf * R2 / 2 + 2 * coupling * sqrt(R2)) / k
end

function davis_yin_ergodic_gap_bound(;
    k, initial, L, M, γ, smooth_solution_gradient_norm=0.0, kwargs...
)
    _require_integer_at_least(k, 0)
    R2 = _require_finite_nonnegative(initial, :initial)
    Lf = _require_finite_positive(L, :L)
    Mf = _require_finite_positive(M, :M)
    γf = _require_finite_positive(γ, :γ)
    cstar = _require_finite_nonnegative(
        smooth_solution_gradient_norm, :smooth_solution_gradient_norm
    )
    margin = 2 / Lf - γf
    known_margin = _known_real(margin)
    known_margin === nothing ||
        known_margin > 0 ||
        throw(DomainError(γ, "requires γ < 2 / L"))
    radius = sqrt(R2)
    numerator = R2 + γf * R2 / margin + 4 * γf * radius * (cstar + Mf)
    return numerator / (2 * γf * (k + 1))
end

"""
    davis_yin_gradient_mapping_bound(; k, initial, L, γ)

Args: iteration count, initial distance, smoothness, and stepsize.
Returns: scalar bound.
- Evaluates the Davis-Yin gradient-mapping guarantee.
"""
davis_yin_gradient_mapping_bound

"""
    davis_yin_stepsize(; L)

Args: smoothness constant.
Returns: scalar stepsize.
- Gives the default Davis-Yin splitting stepsize.
"""
davis_yin_stepsize

"""
    davis_yin_ergodic_gap_bound(; k, initial, L, M, γ)

Args: iteration count, initial distance, smoothness, Lipschitz constant, and stepsize.
Returns: scalar bound.
- Evaluates the Davis-Yin ergodic gap guarantee.
"""
davis_yin_ergodic_gap_bound
