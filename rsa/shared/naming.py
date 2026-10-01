"""Output naming for searchlight RSA runs. The model normalization is always part of the name."""

MODEL_NORMS = ("zscore", "center")
DEFAULT_MODEL_NORM = "center"


def add_model_norm_arg(parser):
    parser.add_argument(
        "--model-norm", choices=MODEL_NORMS, default=DEFAULT_MODEL_NORM, dest="model_norm",
        help="Model embedding normalization per run: zscore = z-score each dimension, "
             "center = subtract the per-dimension mean only. fMRI is always z-scored.")


def searchlight_config(k, delay_sec, bin_sec, skip_sec, method, model_norm, hrf=False):
    delay = "hrf" if hrf else f"delay{int(delay_sec)}s"
    return f"k{k}_{delay}_bin{int(bin_sec)}s_skip{int(skip_sec)}s_{method}_{model_norm}"


def searchlight_stem(fmri_tag, k, delay_sec, bin_sec, skip_sec, method, model_norm):
    return f"rsa_59k_{fmri_tag}_" + searchlight_config(k, delay_sec, bin_sec, skip_sec, method, model_norm)


def method_label(method, model_norm):
    """Comparator label used in group-level names; crossnobis (rho_a) does not depend on the model normalization."""
    return method if method == "rho_a" else f"{method}_{model_norm}"
