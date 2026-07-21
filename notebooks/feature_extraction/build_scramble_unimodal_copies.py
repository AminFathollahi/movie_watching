"""
notebooks/feature_extraction/build_scramble_unimodal_copies.py
=================================================================
Move 3 support script -- omni3b/topoomni's TRUE unimodal "_a"/"_v" embeddings
(from omni3b_extract_intact.py / topo_omni_extract_intact.py) come from
forward passes with NO cross-modal tokens present at all, so they cannot
depend on which audio was paired with which video. This means the nuisance
regressors for the Move-3 scrambled integration contrast can be built by
pure reindexing -- no GPU inference needed -- as long as the SAME
permutation (seed=42, applied to the SAME natsorted 626-segment list) is
used as in omni3b_extract_scramble.py / topo_omni_extract_scramble.py:

    a_scrambled[i] = a_true[perm[i]]   -- audio actually fed to the joint pass at row i
    v_scrambled[i] = v_true[i]          -- video is never permuted, unchanged

Saves these under the same {model}_avscramble/ directory that the joint
"_av" scramble extraction writes into, so integration_*_avscramble runs
in model_registry.py can use _own_unimodal_integration_run() unmodified.

Run with (no GPU / model env needed, pure numpy):
    conda run --no-capture-output -n movie \
        python "notebooks/feature_extraction/build_scramble_unimodal_copies.py"
"""

from pathlib import Path

import numpy as np
from natsort import natsorted

DATA_BASE       = Path("/home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered")
EMBEDDINGS_BASE = Path("/home/amin/Research/Representation/Movie/outputs/model_embeddings")
BIN_SEC, SKIP_SEC = 5.0, 5.0
# {tag: [layers]} -- nemotron has an extra layer 36 (its true final/native layer).
MODEL_TAG_LAYERS = {
    "omni3b": [9, 18, 27],
    "topoomni": [9, 18, 27],
    "nemotron": [9, 18, 27, 36],
}
SCRAMBLE_SEED = 42


def main():
    dur_int, skip_int = int(BIN_SEC), int(SKIP_SEC)
    chunk_suffix = f"_av_chunks_{dur_int}s" if skip_int == dur_int else f"_av_chunks_{dur_int}s_skip{skip_int}s"
    all_segs = natsorted(list(DATA_BASE.rglob(f"*{chunk_suffix}/*.mp4")), key=lambda p: p.name)
    n = len(all_segs)
    assert n > 0, f"No {BIN_SEC}s segments found under {DATA_BASE}"
    print(f"Found {n} segments (must match the row count of _a.npy/_v.npy).")

    rng = np.random.default_rng(SCRAMBLE_SEED)
    perm = rng.permutation(n)
    n_fixed = int((perm == np.arange(n)).sum())
    print(f"[SCRAMBLE_AV] perm computed (seed={SCRAMBLE_SEED}, {n_fixed} incidental self-pairs) "
          f"-- MUST match omni3b_extract_scramble.py / topo_omni_extract_scramble.py.")

    out_tag = f"bin{dur_int}s_skip{skip_int}s"

    for tag, layers in MODEL_TAG_LAYERS.items():
        for layer in layers:
            base = f"{tag}_layer{layer}_mp"
            a_path = EMBEDDINGS_BASE / base / out_tag / f"{base}_a.npy"
            v_path = EMBEDDINGS_BASE / base / out_tag / f"{base}_v.npy"
            if not a_path.exists() or not v_path.exists():
                print(f"[{base}] SKIP -- missing {a_path if not a_path.exists() else v_path}")
                continue
            a_true = np.load(a_path)
            v_true = np.load(v_path)
            assert a_true.shape[0] == n, f"{a_path}: {a_true.shape[0]} rows != {n} segments"
            assert v_true.shape[0] == n, f"{v_path}: {v_true.shape[0]} rows != {n} segments"

            a_scrambled = a_true[perm]
            v_scrambled = v_true  # never permuted

            out_dir = EMBEDDINGS_BASE / f"{base}_avscramble" / out_tag
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{base}_avscramble_a.npy", a_scrambled)
            np.save(out_dir / f"{base}_avscramble_v.npy", v_scrambled)
            print(f"[{base}_avscramble] saved reindexed _a={a_scrambled.shape} "
                  f"(unchanged) _v={v_scrambled.shape} -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
