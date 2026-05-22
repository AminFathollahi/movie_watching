#!/usr/bin/env python3
"""
Evaluate split-half reliability of searchlight RSA maps across different neighborhood sizes (k).
Streams raw fMRI data to memory, processes halves independently, and computes reliability.

Usage:
  conda activate analysis
  python kreliability.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --subjects-list /home/amin/Research/Representation/Movie/data/subjects.txt \
    --timing-csv /home/amin/Research/Representation/Movie/data/segmented_stimulus/filtered/movie_timing.csv \
    --embeddings-dir /home/amin/Research/Representation/Movie/outputs/model_embeddings \
    --template-cifti /home/amin/Research/Representation/Movie/data/average_sub/sg_psc/group_average_sg_psc_cortex_59k.dtseries.nii \
    --indiv-surf-template "/home/amin/Research/Representation/Movie/data/midthickness_1.6/{sub}.{hem}.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii" \
    --workbench /opt/workbench/bin_linux64/wb_command \
    --outdir /home/amin/Research/Representation/Movie/outputs/rsa_reliability \
    --model pe-av-small-16-frame --modality av \
    --k 100 150 200 --n-splits 50 --bin-sec 5.0
"""

import argparse
import gc
import json
import logging
import subprocess
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import existing pipeline components
from preprocess_individual import preprocess_subject
from run_searchlight import get_neighbors, run_searchlight
from rsa.shared.rsa_utils import align_and_assert_bins, preprocess_fmri, process_model_embeddings, compute_rdm
from rsa.shared.cifti_io import get_bm_axis, get_cortex_vertex_indices

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Split-half reliability for RSA maps.")
    p.add_argument("--raw-dir", required=False, default=None)
    p.add_argument("--subjects-list", required=True)
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    
    # NEW: Need individual surfaces to generate split-half averages
    p.add_argument("--indiv-surf-template", required=True,
                   help="Template path to individual surfaces. Use {sub} and {hem} (L/R) placeholders. "
                        "e.g., /data/{sub}/MNINonLinear/{sub}.{hem}.midthickness.59k_fs_LR.surf.gii")
    
    p.add_argument("--workbench", required=True)
    p.add_argument("--outdir", required=True)
    
    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True)
    
    p.add_argument("--k", type=int, nargs="+", default=[100, 150, 200], help="List of k values to compare")
    p.add_argument("--n-splits", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-sec", type=float, default=5.0)
    
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--hrf", action="store_true")
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--n-jobs", type=int, default=-1)
    
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--fmri-suffix", default="sg_psc_gsr")
    return p.parse_args()


def generate_half_surface(subjects: list, hem_short: str, args, out_path: Path):
    """
    Calls wb_command to compute the mathematical average of all midthickness
    surfaces for the subjects in this split-half.
    """
    if out_path.exists():
        return
        
    log.info(f"    Generating average {hem_short} surface for split-half ({len(subjects)} subjects)...")
    cmd = [args.workbench, "-surface-average", str(out_path)]
    
    for sub in subjects:
        surf_file = Path(args.indiv_surf_template.format(sub=sub, hem=hem_short))
        if not surf_file.exists():
            raise FileNotFoundError(f"Missing individual surface required for averaging: {surf_file}")
        cmd.extend(["-surf", str(surf_file)])
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"wb_command -surface-average failed:\n{result.stderr}")


def load_or_stream_half(subjects, args):
    """Loads preprocessed continuous maps from disk if available, otherwise streams from raw."""
    running_mean = None
    n_loaded = 0
    shared_run_trs = None 

    for sub in tqdm(subjects, desc="Processing fMRI Half", leave=False):
        try:
            if args.preprocessed_dir:
                cifti_path = Path(args.preprocessed_dir) / f"{sub}_{args.fmri_suffix}_cortex_59k.dtseries.nii"
                trs_path = Path(args.preprocessed_dir) / f"{sub}_{args.fmri_suffix}_run_trs.npy"
                
                img = nib.load(str(cifti_path))
                data = img.get_fdata(dtype=np.float32).T
                run_trs = np.load(str(trs_path))
            else:
                prep_args = types.SimpleNamespace(sg_filter=True, psc=True, gsr=True)
                data, _, run_trs = preprocess_subject(sub, Path(args.raw_dir), args.tr, prep_args)

            if running_mean is None:
                running_mean = np.zeros_like(data, dtype=np.float64)
                running_mean[:] = data[:]
                shared_run_trs = run_trs
                n_loaded = 1
            else:
                n_loaded += 1
                # MEMORY FIX: In-place sequential calculation prevents a 3.1GB 
                # temporary array allocation during the arithmetic step.
                delta = data - running_mean
                delta /= n_loaded
                running_mean += delta
                del delta
            
            del data
            gc.collect()
        except Exception as e:
            log.warning(f"Failed processing {sub}: {e}")
            
    if running_mean is None:
        raise RuntimeError("No subjects successfully processed for this half.")
    
    return running_mean.astype(np.float32), shared_run_trs


def compute_rsa_map_for_ks(fmri_binned, model_rdm, subjects, split_id, half_label, args):
    """Compute searchlight maps across multiple K values using a half-specific surface."""
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)
    
    cache_dir = Path(args.outdir) / "_geodesic_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    maps_by_k = {k: np.zeros(fmri_binned.shape[1], dtype=np.float32) for k in args.k}
    max_k = max(args.k)
    
    for hem, surf_indices in [("left", left_indices), ("right", right_indices)]:
        hem_short = "L" if hem == "left" else "R"
        surf_indices = surf_indices.astype(np.int32)
        vertex_to_col = np.full(surf_indices.max() + 1, -1, dtype=np.int32) 
        vertex_to_col[surf_indices] = np.arange(len(surf_indices), dtype=np.int32)
        
        fmri_hem = fmri_binned[:, :n_left] if hem == "left" else fmri_binned[:, n_left:]
        offset = 0 if hem == "left" else n_left

        # 1. Generate the half-specific average surface
        half_surf_path = cache_dir / f"{split_id}_{half_label}_{hem_short}_avg.surf.gii"
        generate_half_surface(subjects, hem_short, args, half_surf_path)
        
        # 2. SPEED FIX: Extract max_k neighbors once.
        # Passing sub_id != "group_average" triggers run_searchlight.py's internal
        # logic to instantly delete the 14GB dconn file as soon as this returns.
        sub_id = f"{split_id}_{half_label}"
        neighbors_max = get_neighbors(
            str(half_surf_path), args.workbench, sub_id, hem, max_k, cache_dir
        )
        
        # 3. Clean up the half-specific surface to save disk space
        if half_surf_path.exists():
            half_surf_path.unlink()
        
        # 4. Iterate over requested ks by simply slicing the max array
        for k in sorted(args.k, reverse=True):
            neighbors_k = neighbors_max[:, :k]
            
            corr_hem = run_searchlight(
                fmri_hem, model_rdm, neighbors_k,
                surface_indices=surf_indices, vertex_to_col=vertex_to_col,
                method=args.method, n_jobs=args.n_jobs
            )
            maps_by_k[k][offset: offset + len(corr_hem)] = corr_hem
            
            del corr_hem
            gc.collect()

        del neighbors_max
        gc.collect()

    return maps_by_k


def evaluate_physical_neighborhood(args, cache_dir, n_seeds=1000):
    # (Leaving this intact, but note it currently relies on "group_average_..._geodesic.dconn.nii" 
    # which we are no longer generating here since we generate split-half specific ones. 
    # If you want this to work, it will need a static group dconn generated externally.)
    pass


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = outdir / "split_half_reliability_results.json"
    results = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            results = json.load(f)

    subjects = [ln.strip() for ln in Path(args.subjects_list).read_text().splitlines() if ln.strip()]
    timing_df = pd.read_csv(args.timing_csv)
    
    bin_sec_int = int(args.bin_sec)
    emb_file = Path(args.embeddings_dir) / args.model / f"{bin_sec_int}s" / f"{args.model}_{args.modality}.npy"
    
    model_rdm = None
    model_binned = None
    
    rng = np.random.default_rng(args.seed)
    
    for split_idx in range(args.n_splits):
        split_id = f"split_{split_idx:03d}"
        if split_id in results and all(str(k) in results[split_id] for k in args.k):
            log.info(f"Skipping {split_id} (already computed).")
            continue
            
        log.info(f"--- Starting {split_id} ---")
        rng.shuffle(subjects)
        half_n = len(subjects) // 2
        half1_subs, half2_subs = subjects[:half_n], subjects[half_n:2*half_n]
        
        # --- Half 1 ---
        fmri_h1, run_trs_h1 = load_or_stream_half(half1_subs, args)
        
        # LAZY INIT MODEL RDM: Now we have run_trs_h1, so we can process the model
        if model_rdm is None:
            log.info("Processing model embeddings...")
            model_binned = process_model_embeddings(
                str(emb_file), timing_df, args.bin_sec, args.hrf, args.tr, run_trs_h1, args.delay_sec
            )
            model_rdm = compute_rdm(model_binned.astype(np.float64), method=args.method)

        fmri_binned_h1 = preprocess_fmri(fmri_h1, timing_df, run_trs_h1, args.bin_sec, args.tr, args.delay_sec)
    
    

    
    for split_idx in range(args.n_splits):
        split_id = f"split_{split_idx:03d}"
        if split_id in results and all(str(k) in results[split_id] for k in args.k):
            log.info(f"Skipping {split_id} (already computed).")
            continue
            
        log.info(f"--- Starting {split_id} ---")
        rng.shuffle(subjects)
        half_n = len(subjects) // 2
        half1_subs, half2_subs = subjects[:half_n], subjects[half_n:2*half_n]
        
        # --- Half 1 ---
        fmri_h1, run_trs_h1 = load_or_stream_half(half1_subs, args)
        fmri_binned_h1 = preprocess_fmri(fmri_h1, timing_df, run_trs_h1, args.bin_sec, args.tr, args.delay_sec)
        fmri_binned_h1, _ = align_and_assert_bins(fmri_binned_h1, model_binned)
        maps_h1 = compute_rsa_map_for_ks(fmri_binned_h1, model_rdm, half1_subs, split_id, "half1", args)
        
        del fmri_h1, fmri_binned_h1
        gc.collect()
        
        # --- Half 2 ---
        fmri_h2, run_trs_h2 = load_or_stream_half(half2_subs, args)
        fmri_binned_h2 = preprocess_fmri(fmri_h2, timing_df, run_trs_h2, args.bin_sec, args.tr, args.delay_sec)
        fmri_binned_h2, _ = align_and_assert_bins(fmri_binned_h2, model_binned)
        maps_h2 = compute_rsa_map_for_ks(fmri_binned_h2, model_rdm, half2_subs, split_id, "half2", args)
        
        del fmri_h2, fmri_binned_h2
        gc.collect()
        
        # Evaluate reproducibility per k
        results[split_id] = {}
        for k in args.k:
            map1, map2 = maps_h1[k], maps_h2[k]
            r_global, _ = pearsonr(map1, map2)
            
            thresh1 = np.percentile(map1, 90)
            thresh2 = np.percentile(map2, 90)
            mask1, mask2 = map1 >= thresh1, map2 >= thresh2
            dice = 2 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))
            
            results[split_id][str(k)] = {"r_global": float(r_global), "dice_top10": float(dice)}
            log.info(f"  k={k:3d} | Global r: {r_global:.4f} | Top-10% Dice: {dice:.4f}")

        with open(metadata_file, "w") as f:
            json.dump(results, f, indent=4)

    log.info("--- All Splits Complete ---")
    for k in args.k:
        r_vals = [results[s][str(k)]["r_global"] for s in results]
        d_vals = [results[s][str(k)]["dice_top10"] for s in results]
        log.info(f"Summary k={k:3d}: Global r = {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}, "
                 f"Dice = {np.mean(d_vals):.4f} ± {np.std(d_vals):.4f}")

if __name__ == "__main__":
    main()