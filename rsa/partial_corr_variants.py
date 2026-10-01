"""Consolidate the three five-map partial-correlation families for PE-AV.

Every scalar is a whole-cortex partial Spearman correlation on RDM upper
triangles -- corr(brain, TARGET | NUISANCE) -- never embedding-space
residualization. Nuisance RDMs are regressed out of both sides via banded
ridge before the Spearman correlation is taken (see rsa/partial_rsa.py).

``from_unimodals`` variant, nuisances AudioMAE(a) + VideoMAEv2-Large(v):

``peav_av_given_audiomae``    corr(brain, peav_av | audiomae)
``peav_av_given_videomae``    corr(brain, peav_av | videomae)
``peav_av_given_both``        corr(brain, peav_av | audiomae, videomae)
``audiomae_given_videomae``   corr(brain, audiomae | videomae)
``videomae_given_audiomae``   corr(brain, videomae | audiomae)

``text_aligned_models`` variant, nuisances WavLM-Large(a) + PE-Core ViT-L/14(v):

``peav_av_given_wavlm``       corr(brain, peav_av | wavlm)
``peav_av_given_pecore``      corr(brain, peav_av | pecore)
``peav_av_given_both``        corr(brain, peav_av | wavlm, pecore)
``wavlm_given_pecore``        corr(brain, wavlm | pecore)
``pecore_given_wavlm``        corr(brain, pecore | wavlm)

``own_unimodal`` variant, nuisances are PE-AV's OWN audio(a) + video(v) streams:

``peav_av_given_peav_a``      corr(brain, peav_av | peav_a)
``peav_av_given_peav_v``      corr(brain, peav_av | peav_v)
``peav_av_given_both``        corr(brain, peav_av | peav_a, peav_v)
``peav_a_given_peav_v``       corr(brain, peav_a | peav_v)
``peav_v_given_peav_a``       corr(brain, peav_v | peav_a)

``own_unimodal``'s ``peav_av_given_both`` layer is not a new fit: it is the
existing integration_pe-av-small-16-frame run (kind="integration"), reused
verbatim from its own group_average/pe-av-small-16-frame_av_INTEGRATION/
directory under the different filename that "integration" kind produces
(``integration_partial_r_searchlight.dscalar.nii`` rather than
``partial_corr_r_searchlight.dscalar.nii``).

The five plain no-nuisance maps (peav_av, audiomae, videomae, wavlm, pecore)
are deliberately excluded from all three files -- they are ordinary RSA, not
partial RSA, and already live in their own
group_average/<model>_<modality>/k.../rsa_..._searchlight.npy directories.

Each scalar is appended independently, so interrupted and repeated runs only
load and add maps that are absent from the destination CIFTI.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.naming import DEFAULT_MODEL_NORM  # noqa: E402
from cifti_io import (  # noqa: E402
    get_combined_map_names,
    load_single_map,
    merge_into_combined,
    save_cifti_map,
)

log = logging.getLogger(__name__)

CONFIG = f"k100_delay5s_bin5s_skip5s_spearman_{os.environ.get('MODEL_NORM', DEFAULT_MODEL_NORM)}"
SEARCHLIGHT_FILENAME = "partial_corr_r_searchlight.dscalar.nii"
# kind="integration" runs (rsa/partial_rsa.py) save under this filename instead;
# only the reused peav_av_given_both layer of own_unimodal needs it.
INTEGRATION_SEARCHLIGHT_FILENAME = "integration_partial_r_searchlight.dscalar.nii"

# variant -> ordered list of (source directory under <rsa_root>/, scalar name)
VARIANTS: dict[str, list[tuple[str, str]]] = {
    "from_unimodals": [
        ("pe-av-small-16-frame_av_partial_corr_audiomae_only", "peav_av_given_audiomae"),
        ("pe-av-small-16-frame_av_partial_corr_videomae_only", "peav_av_given_videomae"),
        ("pe-av-small-16-frame_av_partial_corr", "peav_av_given_both"),
        ("audiomae_a_partial_corr_given_videomae", "audiomae_given_videomae"),
        ("videomaev2-large_v_partial_corr_given_audiomae", "videomae_given_audiomae"),
    ],
    "text_aligned_models": [
        ("pe-av-small-16-frame_av_partial_corr_wavlm_only", "peav_av_given_wavlm"),
        ("pe-av-small-16-frame_av_partial_corr_pecore_only", "peav_av_given_pecore"),
        ("pe-av-small-16-frame_av_partial_corr_wavlm_pecore_both", "peav_av_given_both"),
        ("wavlm-large_a_partial_corr_given_pecore", "wavlm_given_pecore"),
        ("pe-core-l14_v_partial_corr_given_wavlm", "pecore_given_wavlm"),
    ],
    "own_unimodal": [
        ("pe-av-small-16-frame_av_partial_corr_own_a_only", "peav_av_given_peav_a"),
        ("pe-av-small-16-frame_av_partial_corr_own_v_only", "peav_av_given_peav_v"),
        ("pe-av-small-16-frame_av_INTEGRATION", "peav_av_given_both"),
        ("pe-av-small-16-frame_a_partial_corr_given_own_v", "peav_a_given_peav_v"),
        ("pe-av-small-16-frame_v_partial_corr_given_own_a", "peav_v_given_peav_a"),
    ],
}

MAP_ORDER: dict[str, tuple[str, ...]] = {
    variant: tuple(name for _, name in sources) for variant, sources in VARIANTS.items()
}


def _searchlight_filename(dir_name: str) -> str:
    """kind="integration" run directories use a different filename (see module docstring)."""
    return (INTEGRATION_SEARCHLIGHT_FILENAME if dir_name.endswith("_INTEGRATION")
            else SEARCHLIGHT_FILENAME)


def source_paths(rsa_root: Path, variant: str) -> dict[str, Path]:
    """Return {scalar_name: source .dscalar.nii path} for one variant."""
    return {
        scalar_name: rsa_root / dir_name / CONFIG / _searchlight_filename(dir_name)
        for dir_name, scalar_name in VARIANTS[variant]
    }


def output_path(rsa_root: Path, variant: str) -> Path:
    """Return the consolidated CIFTI path for one variant."""
    return rsa_root / "pe-av-small-16-frame_av" / (
        f"rsa_59k_raw_{CONFIG}_partial_corr_{variant}.dscalar.nii"
    )


def missing_map_names(destination: Path, variant: str) -> list[str]:
    """Return scalars absent from an existing or new CIFTI, in MAP_ORDER."""
    existing = set(get_combined_map_names(destination))
    return [name for name in MAP_ORDER[variant] if name not in existing]


def append_partial_corr_variant(
    *,
    variant: str,
    rsa_root: Path,
    template_cifti: Path,
) -> list[str]:
    """Append only missing scalars for one variant and return their names."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(VARIANTS)}")

    destination = output_path(rsa_root, variant)
    maps_needed = missing_map_names(destination, variant)
    for map_name in MAP_ORDER[variant]:
        if map_name not in maps_needed:
            log.info("  %-24s already present; skipping", map_name)
    if not maps_needed:
        log.info("All %s maps already exist; nothing to do", variant)
        return []

    sources = source_paths(rsa_root, variant)
    absent = [sources[name] for name in maps_needed if not sources[name].is_file()]
    if absent:
        detail = "\n".join(f"  {path}" for path in absent)
        raise FileNotFoundError(f"Missing partial-corr source map(s):\n{detail}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    for map_name in maps_needed:
        source = sources[map_name]
        data = load_single_map(source)
        merge_into_combined(
            np.asarray(data, dtype=np.float32),
            map_name,
            destination,
            str(template_cifti),
        )
        log.info("  %-24s <- %s", map_name, source)
    return maps_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    parser.add_argument("--rsa-root", required=True)
    parser.add_argument("--template-cifti", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    added = append_partial_corr_variant(
        variant=args.variant,
        rsa_root=Path(args.rsa_root),
        template_cifti=Path(args.template_cifti),
    )
    log.info("Added %d partial-corr maps for variant %s", len(added), args.variant)


def demo():
    template = "/home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii"
    with tempfile.TemporaryDirectory(
        dir="/tmp/claude-1000/-home-amin-Research-Representation-Movie-movie-watching/63cd2761-6f15-4454-9bd1-9fda179d8650/scratchpad"
    ) as tmp:
        rsa_root = Path(tmp)
        rng = np.random.default_rng(0)
        for variant, sources in VARIANTS.items():
            for dir_name, scalar_name in sources:
                out_dir = rsa_root / dir_name / CONFIG
                out_dir.mkdir(parents=True, exist_ok=True)
                fake = rng.normal(size=108441).astype(np.float32)
                save_cifti_map(fake, template, str(out_dir / _searchlight_filename(dir_name)),
                               map_name=scalar_name)

        expected_values = {}
        for variant in VARIANTS:
            sources = source_paths(rsa_root, variant)
            expected_values[variant] = {
                name: load_single_map(path) for name, path in sources.items()
            }
            added = append_partial_corr_variant(
                variant=variant, rsa_root=rsa_root, template_cifti=template,
            )
            assert added == list(MAP_ORDER[variant]), f"unexpected append order: {added}"

        for variant in VARIANTS:
            dest = output_path(rsa_root, variant)
            names = get_combined_map_names(dest)
            assert names == list(MAP_ORDER[variant]), (
                f"{variant}: names {names} != {list(MAP_ORDER[variant])}"
            )
            import nibabel as nib
            data = nib.load(str(dest)).get_fdata(dtype=np.float32)
            assert data.shape == (5, 108441), f"{variant}: shape {data.shape}"
            for i, name in enumerate(names):
                assert np.array_equal(data[i], expected_values[variant][name]), (
                    f"{variant}/{name}: values changed on round-trip"
                )
            print(f"[demo] {variant} OK: shape={data.shape} names={names}")

            # idempotence: re-running appends nothing
            added_again = append_partial_corr_variant(
                variant=variant, rsa_root=rsa_root, template_cifti=template,
            )
            assert added_again == [], f"{variant}: re-run should append nothing, got {added_again}"

    print("[demo] all assertions passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
