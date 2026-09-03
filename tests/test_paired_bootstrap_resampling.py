"""
Self-check for the paired-contrast block-bootstrap resampling scheme.

Context (see rsa/scramble_paired_stats.py and rsa/shared/rsa_utils.py::
corrected_2factor_bootstrap): the AV temporal-scramble binding contrast is
diff = rho_intact - rho_scrambled, paired per subject and per temporal block,
because both conditions are computed from the SAME subject's SAME brain data
for that movie block (only the model-side AV pairing differs). That shared
subject/block noise must cancel in the difference.

Production code computes diff_block = intact_block - scrambled_block BEFORE
calling corrected_2factor_bootstrap, so the bootstrap resamples the ALREADY
-PAIRED difference (JOINT resampling: one subject/block draw applies to both
conditions at once, by construction, since only one array exists at that
point). This is correct.

The alternative (INDEPENDENT resampling) -- bootstrapping rho_intact and
rho_scrambled separately with their own subject/block draws, then summing
the two variances -- breaks the cancellation: the shared subject+block noise
is counted twice instead of cancelling, inflating the null variance and
silently destroying power. This is a genuine statistical error, not mere
conservatism.

This script demonstrates the difference on synthetic paired data with known
autocorrelation (a shared per-subject-per-block factor) and a known true
effect, and asserts that joint resampling recovers a roughly-nominal false-
positive rate with usable power, while independent resampling is severely
underpowered for the same effect.

Run directly: python tests/test_paired_bootstrap_resampling.py
Or via pytest: pytest tests/test_paired_bootstrap_resampling.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.shared.rsa_utils import corrected_2factor_bootstrap  # noqa: E402


def _simulate_paired_block_data(rng, n_subs, n_blocks, sigma_common, sigma_subj,
                                 sigma_noise, true_effect):
    """rho_intact/rho_scrambled share a per-subject-per-block factor (`common`,
    e.g. movie-content-driven RSA strength) and a per-subject baseline
    (`subj`) -- both of which must cancel in intact - scrambled. Only the
    per-condition measurement noise and the true effect do not cancel."""
    common = rng.normal(0, sigma_common, size=(n_subs, n_blocks))
    subj = rng.normal(0, sigma_subj, size=(n_subs, 1))
    noise_i = rng.normal(0, sigma_noise, size=(n_subs, n_blocks))
    noise_s = rng.normal(0, sigma_noise, size=(n_subs, n_blocks))
    rho_intact = common + subj + true_effect + noise_i
    rho_scrambled = common + subj + noise_s
    return rho_intact.astype(np.float32), rho_scrambled.astype(np.float32)


def _joint_pvalue(rho_intact_block, rho_scrambled_block, n_boot, rng):
    """Matches production: difference computed FIRST, bootstrap resamples the diff."""
    diff_block = rho_intact_block - rho_scrambled_block
    diff_stack = diff_block.mean(axis=1)
    var_c2f, _, _ = corrected_2factor_bootstrap(
        diff_stack[:, None], diff_block[:, :, None], n_boot=n_boot, rng=rng)
    se = np.sqrt(max(var_c2f[0], 0.0))
    df = max(min(diff_stack.shape[0] - 1, diff_block.shape[1] - 1), 1)
    t = diff_stack.mean() / se if se > 0 else 0.0
    return float(stats.t.sf(abs(t), df=df) * 2.0)


def _independent_pvalue(rho_intact_block, rho_scrambled_block, n_boot, rng):
    """The statistical error under test: bootstrap each condition SEPARATELY
    (independent subject/block draws per condition), then sum the variances
    as if independent. The shared common/subj noise no longer cancels."""
    intact_stack = rho_intact_block.mean(axis=1)
    scrambled_stack = rho_scrambled_block.mean(axis=1)
    var_i, _, _ = corrected_2factor_bootstrap(
        intact_stack[:, None], rho_intact_block[:, :, None], n_boot=n_boot, rng=rng)
    var_s, _, _ = corrected_2factor_bootstrap(
        scrambled_stack[:, None], rho_scrambled_block[:, :, None], n_boot=n_boot, rng=rng)
    var_diff_indep = var_i[0] + var_s[0]
    se = np.sqrt(max(var_diff_indep, 0.0))
    mean_diff = (intact_stack - scrambled_stack).mean()
    df = max(min(intact_stack.shape[0] - 1, rho_intact_block.shape[1] - 1), 1)
    t = mean_diff / se if se > 0 else 0.0
    return float(stats.t.sf(abs(t), df=df) * 2.0)


def _run_sim(n_sims, n_subs, n_blocks, sigma_common, sigma_subj, sigma_noise,
             true_effect, n_boot, seed):
    rng_master = np.random.default_rng(seed)
    p_joint, p_indep = [], []
    for _ in range(n_sims):
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        ri, rs = _simulate_paired_block_data(
            rng, n_subs, n_blocks, sigma_common, sigma_subj, sigma_noise, true_effect)
        p_joint.append(_joint_pvalue(ri, rs, n_boot, rng))
        p_indep.append(_independent_pvalue(ri, rs, n_boot, rng))
    return np.array(p_joint), np.array(p_indep)


def demo():
    # Parameters chosen so sigma_common (shared subject/block noise, must
    # cancel) dominates sigma_noise (per-condition noise, does not cancel) --
    # matching the real PE-AV data, where per-vertex block-to-block variance
    # is ~2 orders of magnitude larger than subject-to-subject variance.
    n_subs, n_blocks = 30, 16
    sigma_common, sigma_subj, sigma_noise = 0.045, 0.005, 0.02
    n_boot, alpha = 200, 0.05

    p_joint0, p_indep0 = _run_sim(
        150, n_subs, n_blocks, sigma_common, sigma_subj, sigma_noise,
        true_effect=0.0, n_boot=n_boot, seed=1)
    fpr_joint = (p_joint0 < alpha).mean()
    fpr_indep = (p_indep0 < alpha).mean()

    p_joint1, p_indep1 = _run_sim(
        150, n_subs, n_blocks, sigma_common, sigma_subj, sigma_noise,
        true_effect=0.004, n_boot=n_boot, seed=2)
    power_joint = (p_joint1 < alpha).mean()
    power_indep = (p_indep1 < alpha).mean()

    print(f"H0 false-positive rate: joint={fpr_joint:.3f}  independent={fpr_indep:.3f}  "
          f"(nominal alpha={alpha})")
    print(f"H1 power:               joint={power_joint:.3f}  independent={power_indep:.3f}")

    # Joint (= production scheme) has a roughly-nominal FPR and non-trivial power.
    assert fpr_joint < 3 * alpha, "joint FPR should be close to nominal alpha"
    assert power_joint > 0.5, "joint scheme should have usable power at this effect size"

    # Independent resampling is a genuine error: it inflates the null variance
    # by double-counting shared subject/block noise that should cancel, so it
    # is severely underpowered relative to joint at the SAME effect size.
    assert fpr_indep <= fpr_joint, "independent scheme must not be less conservative than joint"
    assert power_indep < power_joint - 0.3, (
        "independent resampling of a paired contrast must lose substantial "
        "power relative to joint resampling -- if this fails, the demonstrated "
        "statistical error no longer reproduces"
    )
    print("OK: joint resampling of the paired diff is correct; independent "
          "resampling would silently and severely lose power.")


def test_joint_resampling_outperforms_independent_resampling():
    demo()


if __name__ == "__main__":
    demo()
