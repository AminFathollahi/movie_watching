import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import gamma, zscore, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import nibabel as nib
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# --- Paths ---
os.environ['PATH'] = '/opt/workbench/bin_linux64:' + os.environ['PATH']

BASE_DIR = Path("/home/amin/Research/Representation/Movie")
DATA_DIR = Path("/home/amin/Research/Representation/Movie/data/Setareh")
EMBEDDINGS_BASE = BASE_DIR / "outputs/model_embeddings"
OUTPUT_BASE = BASE_DIR / "outputs/encoding_results" 

# --- Data Files ---
TIMING_FILE = DATA_DIR / "Data/movie_timing.csv"
MAT_FILE_LEFT = DATA_DIR / "HCP Data/notmean_left_Meanfile.mat"
MAT_FILE_RIGHT = DATA_DIR / "HCP Data/notmean_right_Meanfile.mat"

# --- PARAMETERS ---
NORMALIZATION = True       
USE_HRF = True             
TR = 1.0                   
BIN_SEC = 1.0              # <--- Critical: Script uses this to find "1s" folder

# Ridge Parameters
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CV_FOLDS = 4 
CHUNK_SIZE = 200

# --- MODELS CONFIGURATION ---
# Format: "Folder Name": ("Filename Prefix", "Display Label")
models = {
    "pe-av-small-16-frame": ("peav_small_16-frame", "pe-av-small-16-frame")
}

modalities = {
    # Key = Output Folder Name, Value = Suffix in filename
    # "joint": "av", 
    "audio": "a", 
    "video": "v"
}

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def load_model_data(path):
    path = str(path)
    if not os.path.exists(path):
        print(f"  ERROR: File not found at {path}")
        return None

    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.pt'):
        import torch
        data = torch.load(path, map_location='cpu')
        return data.numpy() if hasattr(data, 'numpy') else np.array(data)
    # Fallback
    return np.load(path)

def load_mat_data(mat_path):
    print(f"Loading fMRI: {mat_path}...")
    try:
        mat = sio.loadmat(mat_path)
        key = max(mat.keys(), key=lambda k: mat[k].size if isinstance(mat[k], np.ndarray) else 0)
        data = mat[key]
        if data.shape[0] < data.shape[1]: 
            data = data.T # Ensure (Vertices, Time)
        return data
    except Exception as e:
        print(f"  Error: {e}")
        return None

def extract_fmri_segments(fmri_data, timing_df, tr):
    """Slices fMRI based on CSV timing."""
    print(f"  Slicing fMRI for {len(timing_df)} videos...")
    segments = []
    n_vertices, n_total = fmri_data.shape
    
    total_len = 0
    for _, row in timing_df.iterrows():
        start = int(np.round(row['onset_sec'] / tr))
        dur = int(np.round(row['duration_sec'] / tr))
        end = start + dur
        
        if end > n_total: end = n_total
        seg = fmri_data[:, start:end]
        segments.append(seg)
        total_len += seg.shape[1]
        
    print(f"  Total fMRI timepoints extracted: {total_len}")
    return np.concatenate(segments, axis=1)

def spm_hrf(tr, oversampling=16):
    dt = tr / oversampling
    u = np.arange(0, 32, dt) - dt/2
    h = (6 * gamma.pdf(u, 6, loc=0, scale=1) - 1/6 * gamma.pdf(u, 16, loc=0, scale=1))
    return h / np.sum(h)

def process_model(model_data, timing_df, normalization=True, hrf=True, tr=1.0, bin_sec=1.0):
    """Convolves per video."""
    hrf_kernel = spm_hrf(bin_sec, oversampling=1) if hrf else None
    bins_per_video = (timing_df['duration_sec'] / bin_sec).astype(int).values
    
    # Trim model if slightly longer than CSV total
    total_dur = np.sum(bins_per_video)
    if model_data.shape[0] > total_dur:
        model_data = model_data[:total_dur]
        
    segments = []
    curr = 0
    for n in bins_per_video:
        if n == 0: continue
        vid = model_data[curr : curr + n]
        
        if hrf:
            conv_vid = np.zeros_like(vid)
            for i in range(vid.shape[1]):
                # Convolve full then slice to original length (tail cut)
                conv_vid[:, i] = np.convolve(vid[:, i], hrf_kernel, mode='full')[:n]
            segments.append(conv_vid)
        else:
            segments.append(vid)
        curr += n
        
    full = np.concatenate(segments, axis=0)
    
    if normalization:
        return zscore(full, axis=0)
    return full

def run_ridge(X, Y, alphas, n_splits=4, chunk_size=200):
    """Ridge Regression with Progress Bar"""
    Y_T = Y.T # (Time, Vertices)
    n_v = Y_T.shape[1]
    corrs = np.zeros(n_v)
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    model = RidgeCV(alphas=alphas)
    
    n_done = 0
    # Process in chunks
    for i in range(0, n_v, chunk_size):
        idx = np.arange(i, min(i+chunk_size, n_v))
        y_chunk = Y_T[:, idx]
        
        # Valid mask (non-flat signals)
        valid = np.std(y_chunk, axis=0) > 1e-10
        valid_idx = np.where(valid)[0]
        
        if len(valid_idx) > 0:
            y_valid = y_chunk[:, valid_idx]
            preds = np.zeros_like(y_valid)
            
            for train, test in kf.split(X):
                model.fit(X[train], y_valid[train])
                preds[test] = model.predict(X[test])
            
            # Vectorized Correlation
            real_centered = y_valid - y_valid.mean(axis=0)
            pred_centered = preds - preds.mean(axis=0)
            numer = np.sum(real_centered * pred_centered, axis=0)
            denom = np.sqrt(np.sum(real_centered**2, axis=0) * np.sum(pred_centered**2, axis=0))
            corrs[i + valid_idx] = numer / (denom + 1e-10)
            
        n_done += len(idx)
        # Update progress on same line
        print(f"Progress: {n_done}/{n_v} ({(n_done/n_v)*100:.1f}%)", end='\r')
        
    print("") # Newline when done
    return corrs

def save_gifti(data, path):
    da = nib.gifti.GiftiDataArray(data.astype(np.float32), intent='NIFTI_INTENT_NONE', datatype='NIFTI_TYPE_FLOAT32')
    nib.save(nib.GiftiImage(darrays=[da]), str(path))

# ============================================================================
# 3. MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"--- RIDGE ENCODING (Model Resolution: {int(BIN_SEC)}s) ---")
    
    # 1. Load Timing & fMRI
    df = pd.read_csv(TIMING_FILE)
    print("Loading Left Hemisphere...")
    fL = extract_fmri_segments(load_mat_data(MAT_FILE_LEFT), df, TR)
    print("Loading Right Hemisphere...")
    fR = extract_fmri_segments(load_mat_data(MAT_FILE_RIGHT), df, TR)
    
    if NORMALIZATION:
        print("Normalizing fMRI...")
        fL = np.nan_to_num(zscore(fL, axis=1))
        fR = np.nan_to_num(zscore(fR, axis=1))

    # 2. Loop Models
    # 'folder' is the dir name, 'prefix' is the filename start, 'label' is output folder name
    for folder, (prefix, label) in models.items():
        for mod_key, mod_file_suffix in modalities.items():
            
            # --- PATH CONSTRUCTION ---
            # 1. Subfolder based on BIN_SEC (1.0 -> "1s", 2.0 -> "2s")
            res_folder = f"{int(BIN_SEC)}s" 
            
            # 2. Filename: prefix + "_" + suffix + ".npy"
            # Ex: "peav_small_16-frame" + "_" + "joint_av" + ".npy"
            filename = f"{prefix}_{mod_file_suffix}.npy"
            
            model_path = EMBEDDINGS_BASE / folder / res_folder / filename
            
            print(f"\nTarget Model: {model_path}")
            data = load_model_data(model_path)
            
            if data is None: continue

            # --- PROCESS ---
            print(f"Convolving Model (HRF={USE_HRF})...")
            X = process_model(data, df, normalization=NORMALIZATION, hrf=USE_HRF, tr=TR, bin_sec=BIN_SEC)
            
            # Trim mismatch
            limit = min(X.shape[0], fL.shape[1])
            if X.shape[0] != fL.shape[1]:
                print(f"Trimming data to length {limit}")
                X = X[:limit]
                fL_run = fL[:, :limit]
                fR_run = fR[:, :limit]
            else:
                fL_run, fR_run = fL, fR
            
            out_dir = OUTPUT_BASE /f"{label}_bin{int(BIN_SEC)}s" / mod_key
            out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"--> Encoding Left ({label})...")
            cL = run_ridge(X, fL_run, ALPHAS)
            np.save(out_dir / f"{label}_bin{int(BIN_SEC)}s_{mod_key}_{'normalized' if NORMALIZATION else 'notnormalized'}_{'hrf' if USE_HRF else 'nohrf'}_left.npy", cL)
            save_gifti(cL, out_dir / f"encoding_{label}_bin{int(BIN_SEC)}s_{mod_key}_{'normalized' if NORMALIZATION else 'notnormalized'}_{'hrf' if USE_HRF else 'nohrf'}_left.func.gii")
            
            print(f"--> Encoding Right ({label})...")
            cR = run_ridge(X, fR_run, ALPHAS)
            np.save(out_dir / f"{label}_bin{int(BIN_SEC)}s_{mod_key}_{'normalized' if NORMALIZATION else 'notnormalized'}_{'hrf' if USE_HRF else 'nohrf'}_right.npy", cR)
            save_gifti(cR, out_dir / f"enconding_{label}_bin{int(BIN_SEC)}s_{mod_key}_{'normalized' if NORMALIZATION else 'notnormalized'}_{'hrf' if USE_HRF else 'nohrf'}_right.func.gii")
            
    print("\nDone.")
