using Argo
using Argo.Atoms
using Argo.CurvatureTransfer

A = [1.0 0.0; 0.0 2.0]
y = [1.0, -1.0]
x = variable(:x; shape=[2])
loss = logistic_loss(A, y; mean=true)(x)
penalty = l1(0.1; n=2)(x)
quadratic = 0.01 * sum_squares()(x)
problem = minimize(loss + penalty + quadratic)

rule = CurvatureTransferRule()
found = certificates(
    problem;
    reformulations=Argo.Authoring.AbstractReformulationRule[rule],
    reformulation_depth=1,
)
transferred = filter(
    certificate -> Argo.Authoring.has_reformulation(certificate, :curvature_transfer),
    found,
)

println("curvature-transfer certificates: ", length(transferred))
if !isempty(transferred)
    interval = parameter_interval(first(transferred), :rho)
    println(
        "rho domain: (",
        evaluate_scalar(interval.lower),
        ", ",
        evaluate_scalar(interval.upper),
        ")",
    )
end
