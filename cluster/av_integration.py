"""
cluster/av_integration.py
============================
Principled AV-integration cartography, replacing the earlier "which of a/v/av
has more FDR-sig cells" heuristic (which trivially favors av everywhere,
since a richer joint embedding explains more variance regardless of whether
the brain actually needs correct audiovisual correspondence).

Reuses the CONTROL-CONDITION idea from rsa/topoomni_av_separability_localizer.py
(the repo's own in-silico AV-integration localizer): genuine integration means
the brain cares about the CORRECT pairing of the two streams, not just their
joint richness. So for each network, "real av" state partition is compared
against three controls, each computed by run_cluster.py exactly like a normal
modality (same K, same HMM, same permutation test) using different embeddings:

  avscramble     : real audio + a randomly-permuted video (breaks temporal
                   correspondence — tests binding/synchrony sensitivity)
  clsav_from_a   : real audio + blanked video (tests whether video is needed
                   ON TOP OF audio)
  clsav_from_v   : blanked audio + real video (tests whether audio is needed
                   ON TOP OF video)

A network is "av_integrative" only if real av beats ALL THREE controls in
FDR-significant cell count (Stein & Meredith-style superadditivity: AV
response > every unimodal/mismatched control, not just richer-on-average).

Also cross-references each network against the Glasser MMP atlas (reused
from rsa/glasser.py) to report whether it aligns with a KNOWN named region
or is spread thinly across many (a novel/heterogeneous functional grouping
not captured by an anatomically/multimodally-defined atlas).
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsa"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DATA, OUTPUTS  # noqa: E402
from io_cluster import write_dlabel, get_bm_axis, GROUP_AVG_CIFTI  # noqa: E402
from rsa.glasser import load_glasser_parcels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GLASSER_DLABEL = (f"{DATA}/HCP_S1200_GroupAvg_v1/"
                  "Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors."
                  "59k_fs_LR.dlabel.nii")

CONDITIONS = ("a", "v", "av", "avscramble", "clsav_from_a", "clsav_from_v")
AV_CONTROLS = ("avscramble", "clsav_from_a", "clsav_from_v")

CATEGORY_RGBA = {
    "av_integrative": (0.55, 0.20, 0.75, 1.0),   # purple — beats ALL 3 controls
    "av_nonspecific": (0.90, 0.60, 0.10, 1.0),   # orange — av-shaped wins, but not via binding/both-modality need
    "auditory":       (0.85, 0.20, 0.20, 1.0),   # red
    "visual":         (0.20, 0.40, 0.85, 1.0),   # blue
    "unclassified":   (0.55, 0.55, 0.55, 1.0),   # gray — nothing reached significance anywhere
}


def _model_dir_for(base_model: str, condition: str) -> str:
    """condition in CONDITIONS -> on-disk {model} name used by run_cluster.py."""
    if condition in ("a", "v", "av"):
        return base_model
    return f"{base_model}_{condition}"  # avscramble/clsav_from_a/clsav_from_v are their own on-disk models


def load_condition_interactions(output_dir: Path, base_model: str, config: str):
    """Load interaction_matrix.npy + interaction_pvals_fdr.npy for all 6 conditions."""
    M_by_cond, pvals_by_cond = {}, {}
    for cond in CONDITIONS:
        model_name = _model_dir_for(base_model, cond)
        modality = cond if cond in ("a", "v") else "av"
        run_dir = output_dir / "group_average" / f"{model_name}_{modality}" / config
        M_by_cond[cond] = np.load(str(run_dir / "interaction_matrix.npy"))
        pvals_by_cond[cond] = np.load(str(run_dir / "interaction_pvals_fdr.npy"))
        log.info(f"  {cond:14s} <- {run_dir.name}  M={M_by_cond[cond].shape}")
    return M_by_cond, pvals_by_cond


def classify_networks(M_by_cond: dict, pvals_by_cond: dict) -> dict:
    """Per-network category using the superadditivity-over-controls criterion."""
    n_networks = M_by_cond["av"].shape[1]
    result = {}
    for g in range(n_networks):
        n_sig = {c: int((pvals_by_cond[c][:, g] < 0.05).sum()) for c in CONDITIONS}
        effect = {c: float(np.abs(M_by_cond[c][:, g] - M_by_cond[c][:, g].mean()).mean())
                 for c in CONDITIONS}

        if max(n_sig.values()) == 0:
            category = "unclassified"
        else:
            best_cond = max(CONDITIONS, key=lambda c: (n_sig[c], effect[c]))
            if best_cond == "av" and all(n_sig["av"] > n_sig[ctrl] for ctrl in AV_CONTROLS):
                category = "av_integrative"
            elif best_cond == "av" or best_cond in AV_CONTROLS:
                # av-shaped embedding wins, but real av did not clear every control
                # (co-occurrence-driven or single-modality-driven-through-av, not binding-specific)
                category = "av_nonspecific"
            elif best_cond == "a":
                category = "auditory"
            else:
                category = "visual"

        binding_gain = n_sig["av"] - n_sig["avscramble"]
        result[g] = {
            "category": category, "n_sig": n_sig, "effect": {k: round(v, 4) for k, v in effect.items()},
            "binding_gain": binding_gain,
        }
        log.info(f"  network {g}: category={category}  n_sig={n_sig}  binding_gain={binding_gain}")
    return result


def atlas_crossref(vertex_labels: np.ndarray, template_cifti: str, glasser_dlabel: str = GLASSER_DLABEL,
                   purity_threshold: float = 0.4) -> dict:
    """Per network: dominant Glasser parcel(s) + known-vs-novel verdict."""
    bm_axis = get_bm_axis(template_cifti)
    parcels = load_glasser_parcels(glasser_dlabel, bm_axis)  # {name: fmri-grayordinate indices}

    vertex_parcel = np.full(vertex_labels.shape, "", dtype=object)
    for name, idx in parcels.items():
        vertex_parcel[idx] = name

    result = {}
    for g in sorted(np.unique(vertex_labels[vertex_labels != -1])):
        members = vertex_parcel[vertex_labels == g]
        members = members[members != ""]
        n_total = int((vertex_labels == g).sum())
        counts = Counter(members)
        top3 = counts.most_common(3)
        top3_frac = [(name, round(count / n_total, 3)) for name, count in top3]
        purity = sum(f for _, f in top3_frac)
        n_parcels_touched = len(counts)

        if top3_frac and top3_frac[0][1] >= purity_threshold:
            verdict = f"known:{top3_frac[0][0]}"
        else:
            verdict = f"novel/mixed({n_parcels_touched} parcels, top={top3_frac[0][1] if top3_frac else 0.0:.2f})"

        result[int(g)] = {
            "n_vertices": n_total, "top3_parcels": top3_frac, "purity_top3": round(purity, 3),
            "n_parcels_touched": n_parcels_touched, "verdict": verdict,
        }
        log.info(f"  network {g}: {verdict}  (top3={top3_frac})")
    return result


def build_category_dlabel(vertex_labels: np.ndarray, classification: dict, atlas: dict,
                          template_cifti: str, out_path: str):
    networks_sorted = sorted(classification.keys())
    label_names, network_rgba = {0: "unassigned"}, {}
    for g in networks_sorted:
        key = g + 1
        cat = classification[g]["category"]
        verdict = atlas[g]["verdict"].split("(")[0]  # keep short in the label name
        label_names[key] = f"net{g}_{cat}_{verdict}"
        network_rgba[key] = CATEGORY_RGBA[cat]
    write_dlabel(vertex_labels, template_cifti, out_path, network_rgba=network_rgba,
                label_names=label_names, map_name="av_integration_category")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", default=str(OUTPUTS / "cluster"))
    p.add_argument("--template-cifti", default=GROUP_AVG_CIFTI)
    p.add_argument("--glasser-dlabel", default=GLASSER_DLABEL)
    p.add_argument("--model", required=True, help="base model name, e.g. nemotron_layer27_mp")
    p.add_argument("--config", required=True)
    p.add_argument("--spatial-config", required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    spatial_dir = output_dir / "group_average" / "_spatial" / args.spatial_config
    vertex_labels = np.load(str(spatial_dir / "spatial_vertex_labels.npy"))
    log.info(f"vertex_labels: {vertex_labels.shape}  n_networks={len(np.unique(vertex_labels[vertex_labels!=-1]))}")

    M_by_cond, pvals_by_cond = load_condition_interactions(output_dir, args.model, args.config)
    classification = classify_networks(M_by_cond, pvals_by_cond)
    atlas = atlas_crossref(vertex_labels, args.template_cifti, args.glasser_dlabel)

    out_dir = output_dir / "group_average" / f"{args.model}_av_integration" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    dlabel_path = out_dir / "vertex_av_integration_category.dlabel.nii"
    build_category_dlabel(vertex_labels, classification, atlas, args.template_cifti, str(dlabel_path))

    modality_code = np.full(vertex_labels.shape, -1, dtype=np.int32)
    cat_to_code = {c: i for i, c in enumerate(CATEGORY_RGBA)}
    for g in classification:
        modality_code[vertex_labels == g] = cat_to_code[classification[g]["category"]]
    np.save(str(out_dir / "vertex_av_integration_code.npy"), modality_code)

    report = {g: {**classification[g], "atlas": atlas[g]} for g in classification}
    (out_dir / "network_av_integration_report.json").write_text(json.dumps(report, indent=2))

    cat_counts = Counter(classification[g]["category"] for g in classification)
    log.info(f"Saved: {dlabel_path}")
    log.info(f"SUMMARY model={args.model}: n_networks={len(classification)}  category_counts={dict(cat_counts)}")


def demo():
    """Synthetic check: a network with a genuinely planted binding effect
    (av >> all 3 controls) must be classified av_integrative; a network
    where avscramble matches av must NOT be (co-occurrence, not binding)."""
    K, G = 10, 3
    M_by_cond, pvals_by_cond = {}, {}
    for cond in CONDITIONS:
        M_by_cond[cond] = np.random.default_rng(0).normal(size=(K, G))
        pvals_by_cond[cond] = np.ones((K, G))  # nothing sig by default

    # Network 0: genuine AV integration — only real av clears significance.
    pvals_by_cond["av"][:3, 0] = 0.01
    # Network 1: av-shaped wins but scramble matches it too -> not integrative.
    pvals_by_cond["av"][:3, 1] = 0.01
    pvals_by_cond["avscramble"][:3, 1] = 0.01
    # Network 2: nothing significant anywhere -> unclassified.

    result = classify_networks(M_by_cond, pvals_by_cond)
    assert result[0]["category"] == "av_integrative", result[0]
    assert result[1]["category"] != "av_integrative", result[1]
    assert result[2]["category"] == "unclassified", result[2]
    print(f"[demo] classify_networks OK: {[(g, result[g]['category']) for g in result]}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        demo()
    else:
        main()
