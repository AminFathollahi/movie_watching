# Validation of 200 LBOEs

**Theory.** Laplace-Beltrami eigenfunctions form a coarse-to-fine spatial basis;
more modes preserve finer connectivity structure. Banded ridge regularization is
selected by leave-one-run-out CV, so overfitting is judged on held-out movie TRs,
not by basis count alone.

| Requested | Actual A / P | Mean full R² | Full-map r vs 200 |
|---:|---:|---:|---:|
| 50 | 50 / 50 | 0.253867 | 0.99629 |
| 100 | 100 / 100 | 0.259799 | 0.99851 |
| 200 | 200 / 126 | 0.263470 | 1.00000 |

**Empirical result.** The capped-200 model has the highest mean held-out full-model
R², so the sensitivity analysis provides no evidence that the larger spatial basis
overfits. The requested count is a per-ROI, shared-hemisphere maximum:
`min(requested, n_L−2, n_R−2)`. The posterior top-1% PE-AV ROI has **128 left +
242 right = 370 bilateral vertices**, so its limiting left hemisphere yields
`128−2 = 126` LBOEs. The two-mode margin keeps SciPy's sparse eigensolver safely
below the graph dimension and away from its degenerate numerical boundary.
