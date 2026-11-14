"""Output naming for searchlight RSA runs. The brain RDM and the model normalization are always part of the name."""

from pathlib import Path

MODEL_NORMS = ("zscore", "center")
DEFAULT_MODEL_NORM = "center"


def add_model_norm_arg(parser):
    parser.add_argument(
        "--model-norm", choices=MODEL_NORMS, default=DEFAULT_MODEL_NORM, dest="model_norm",
        help="Model embedding normalization per run: zscore = z-score each dimension, "
             "center = subtract the per-dimension mean only. fMRI is always z-scored.")


BRAIN_RDMS = {"correlation": "pearson", "euclidean": "euclidean"}


def rsa_method(method, distance="correlation"):
    """Brain RDM and comparator: pearson_spearman for the correlation distance (1 - Pearson r) with Spearman."""
    return f"{BRAIN_RDMS[distance]}_{method}"


def searchlight_config(k, delay_sec, bin_sec, skip_sec, method, model_norm, hrf=False):
    delay = "hrf" if hrf else f"delay{int(delay_sec)}s"
    return f"k{k}_{delay}_bin{int(bin_sec)}s_skip{int(skip_sec)}s_{method}_{model_norm}"


def searchlight_stem(fmri_tag, k, delay_sec, bin_sec, skip_sec, method, model_norm):
    return f"rsa_59k_{fmri_tag}_" + searchlight_config(k, delay_sec, bin_sec, skip_sec, method, model_norm)


def method_label(method, model_norm):
    """Comparator label used in group-level names; crossnobis (rho_a) does not depend on the model normalization."""
    return method if method == "rho_a" else f"{rsa_method(method)}_{model_norm}"


def single_model(modality):
    """Name tag and map name of a searchlight.py run on one embedding: joint_only and rho_j for av."""
    return ("joint_only", "rho_j") if modality == "av" else (f"{modality}_only", f"rho_{modality}")


def searchlight_path(root, model, modality, stem):
    """searchlight.py output of one model under root ({output}/group_average or {output}/subject_data/{subject})."""
    return Path(root) / model / f"{stem}_{single_model(modality)[0]}_models.dscalar.nii"
