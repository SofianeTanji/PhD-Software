# Algorithm

## Objective and feature map

`SnacksSVM` fits the regularized hinge-loss objective

\[
F(u) = \frac{1}{n}\sum_{i=1}^n \max(0, 1-y_i z_i^T u)
       + \frac{\lambda}{2}\lVert u\rVert_2^2,
\]

where labels are encoded as \(-1\) and \(+1\), and \(z_i\) is the Nyström
feature vector of a training sample. The estimator does not fit a separate
intercept.

The feature map samples `m` landmark rows uniformly without replacement. It
forms their kernel matrix, adds `ridge_mu` to its diagonal, and computes its
eigendecomposition. Positive eigenvalues are retained and the kernel values
against the landmarks are multiplied by the eigenvectors scaled by inverse
square-root eigenvalues. Training and prediction use this same fitted map.
The estimator supports RBF, linear, and callable kernels and uses float32
feature matrices and solver weights.

## Restarted regularized stages

`SnacksSVM.fit` calls `rassg_r`. Each stage runs stochastic subgradient updates
with an additional centering penalty
\(\lVert u-c\rVert_2^2/(2\beta)\), where \(c\) is the preceding stage's output.
The stage averages selected iterates from the final portion of its trajectory
and uses that average as the next center. By default, 16 positions are sampled
with replacement from the last half of the stage; duplicate positions are
counted once in the average.

At restart index \(r=0,1,\ldots\), the inner-stage length is the rounded value of
`m_inner0 * growth**r`. Each restart resets
`beta = beta0_scale / (lam * sqrt(m_inner))`; each subsequent stage divides
`beta` by `beta_decay`. Restarts warm-start from the preceding restart's last
stage.

| Parameter | Default | Meaning |
|---|---:|---|
| `lam` | 0.001 | L2 regularization strength |
| `m` | 100 | Nyström landmark count |
| `ridge_mu` | 0.000001 | Landmark-kernel diagonal regularization |
| `restarts` | 4 | Number of outer restarts |
| `stages_per_restart` | 10 | Number of stages per restart |
| `m_inner0` | 512 | Inner updates per stage in the first restart |
| `growth` | 2.0 | Growth of stage length between restarts |
| `beta0_scale` | 10.0 | Initial centering-parameter scale |
| `beta_decay` | 1.35 | Within-restart centering-parameter decay |
| `val_ratio` | 0.15 | Training fraction reserved for stage selection |

## Selection and prediction

With validation enabled, the landmarks and optimizer use the remaining training
samples. All configured stages run, and the estimator selects the stage with
the highest validation accuracy. Without validation (`val_ratio=0`), it returns
the final stage. This is stage selection rather than early termination.

`decision_function(X)` evaluates the fitted feature map and returns its product
with the selected weight vector. `predict(X)` thresholds these scores at zero
and maps them back to the original two label values.

The low-level module also exposes ASSG-c and ASSG-r variants. The L1 experiment
scripts call a separate `rassg_r_l1` solver; `SnacksSVM` uses the L2 objective
above.
