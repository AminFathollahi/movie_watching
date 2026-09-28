"""
cf_modeling/lib/cf_model.py
=============================
CfModel — extends vicsompy's MssCf for HCP group-average CF analysis on
CIFTI grayordinate data, without lookup-table splicing.

Architecture
------------
Vicsompy's MssCf expects:
  1. Surface data in full-sphere space (118584 vertices for 59k_fs_LR), with
     zeros at medial-wall positions.
  2. Pre-built Subsurface objects loaded from {modality}_subsurface.pickle.
  3. A YAML config with all required sub-dicts.
  4. A subject object with .subject (str) and .out_csv (directory path).

CfModel wraps all of this:
  * inject_subsurfaces()     — load pre-built subsurfaces directly, bypassing
                               make_subsurfaces() and the expensive geodesic
                               distance step (create()).
  * make_dm_grayord()        — build design matrix from CIFTI grayordinate data
                               (59412 verts) after internal sphere-space conversion.
  * fit_grayord()            — fit model; Y targets stay in grayordinate space.
  * test_xval_grayord()      — evaluate on test split; returns xval_score and
                               test_split_scores in grayordinate (59412,) shape.
  * compute_null_r2()        — OLS null model from ROI mean timecourses, matching
                               vicsompy's test_null_model() math exactly.
  * save_all_maps()          — write every R² map as npy and (optionally) CIFTI.

What is NOT done (intentionally)
---------------------------------
  * No splicing: load_lookups(), splice_lookups(), save_spliced_lookups() are
    not called.  These require lookup-table CSV files that we do not generate
    for custom ROI pairs.
  * No prepare_frame() / save_frame(): our maps are saved directly as npy and
    CIFTI via save_all_maps().

Usage (see 02_fit_cf_model.py)
-------------------------------
    nm = CfModel(subject_adapter, analysis_name, yaml=temp_yaml_path)
    nm.inject_subsurfaces([sub_a, sub_b], [roi_a, roi_b], [n_lboe_a, n_lboe_b])
    nm.make_dm_grayord(train_data, bm_axis)        # (59412, T_train)
    nm.prep_pipeline(run_durations=run_onsets_arr)  # LORO-CV onsets
    nm.fit_grayord(train_data)                      # Y targets: (T_train, 59412)
    nm.get_params()                                 # betas, train split scores, alphas
    nm.test_xval_grayord(test_data, bm_axis)        # R² on test set
    null_r2 = nm.compute_null_r2(test_data, bm_axis)  # OLS null baseline
    nm.save_outcomes()                              # npy files via MssCf.saveout()
    nm.save_all_maps(null_r2, roi_a, roi_b, bm_axis, template_cifti, out_dir)
"""

import logging
import os
import sys

import numpy as np

try:
    from cf_modeling.cf_naming import cf_model_map_stems
except ModuleNotFoundError:
    from cf_naming import cf_model_map_stems

# ---------------------------------------------------------------------------
# Vicsompy import — resolved at runtime from the source repo.
# The VICSOMPY_REPO path is injected by 01/02 scripts before this module
# is imported.  We guard with a try/except to give a clear error message.
# ---------------------------------------------------------------------------
try:
    from vicsompy.modeling import MssCf  # noqa: E402 (import after sys.path)
except ImportError as exc:
    raise ImportError(
        "Cannot import vicsompy.modeling or himalaya.  "
        "Ensure VICSOMPY_REPO is on sys.path before importing cf_model.\n"
        f"Original error: {exc}"
    ) from exc

from .data_adapter import grayord_to_sphere_space

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone null-model helper (mirrors vicsompy test_null_model math)
# ---------------------------------------------------------------------------

def _fit_null_r2(regressor: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """OLS null model: R² for all targets.

    Model: Y ~ intercept + beta * regressor
    Mathematically identical to vicsompy's test_null_model():
        dm = column_stack([v, ones(T)])
        betas = lstsq(dm, Y.T)[0]
        yhat = dot(dm, betas).T
        null_rsq = 1 - (Y - yhat).var(-1) / Y.var(-1)

    Parameters
    ----------
    regressor : (T,) float — mean timecourse of one ROI on the test set.
    Y         : (T, n_targets) float — test targets (grayordinate, z-scored).

    Returns
    -------
    R2_null : (n_targets,) float32
    """
    T = Y.shape[0]
    dm = np.column_stack([regressor, np.ones(T)])
    betas, _, _, _ = np.linalg.lstsq(dm, Y, rcond=None)
    Y_hat = dm @ betas
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    R2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return R2.astype(np.float32)


# ---------------------------------------------------------------------------
# CfModel
# ---------------------------------------------------------------------------

class CfModel(MssCf):
    """Extended MssCf for CIFTI grayordinate targets, without splicing.

    Inherits all of MssCf's pipeline-building, fitting, and scoring logic.
    Overrides or adds data-handling methods for our setup.
    """

    # ------------------------------------------------------------------
    # Subsurface injection (replaces make_subsurfaces)
    # ------------------------------------------------------------------

    def inject_subsurfaces(
        self,
        subsurfaces: list,
        roi_names: list,
        n_lboes: list,
    ) -> None:
        """Inject pre-built Subsurface objects, bypassing make_subsurfaces().

        This avoids the expensive geodesic-distance step in create() and the
        need for CSV masks on disk at fit time (masks are only needed if
        building subsurfaces from scratch via make_subsurfaces(force_new=True)).

        After calling this method, make_dm_grayord() and test_xval_grayord()
        can be called directly.

        Parameters
        ----------
        subsurfaces : list of Subsurface — pre-built with LBOEs
                      (from 01_extract_geometry.py → pkl cache).
        roi_names   : list of str — modality names in the same order as subsurfaces.
                      Must match the YAML's dm.modalities list.
        n_lboes     : list of int — number of LBOEs per ROI (may be < N_LBOE_MAX
                      if the ROI has fewer vertices).
        """
        if len(subsurfaces) != len(roi_names) or len(subsurfaces) != len(n_lboes):
            raise ValueError(
                f"subsurfaces, roi_names, n_lboes must all have the same length. "
                f"Got {len(subsurfaces)}, {len(roi_names)}, {len(n_lboes)}."
            )

        # Override the YAML-loaded modalities with our explicit list.
        self.modalities = list(roi_names)
        self.subsurfaces = list(subsurfaces)

        # comps_per_hem[i] = n_lboes[i], mirroring define_modality_indices().
        self.comps_per_hem = list(n_lboes)

        # Replicate define_modality_indices() slice computation.
        # Each modality contributes 2 * n_lboe columns (L + R hemispheres).
        vals = np.insert(np.cumsum(np.array(n_lboes, dtype=int) * 2), 0, 0)
        self.modality_idxs = [
            np.arange(vals[i], vals[i + 1]) for i in range(len(n_lboes))
        ]
        log.info(
            "Subsurfaces injected: %s (n_lboes=%s, band_sizes=%s)",
            roi_names,
            n_lboes,
            [2 * n for n in n_lboes],
        )

    # ------------------------------------------------------------------
    # Design matrix from grayordinate data
    # ------------------------------------------------------------------

    def make_dm_grayord(
        self,
        grayord_train: np.ndarray,
        bm_axis,
    ) -> None:
        """Build the design matrix from CIFTI grayordinate training data.

        Converts grayord_train to sphere space internally, then calls the
        parent MssCf.make_dm() which indexes data[subsurface_verts, :] and
        projects onto LBOEs.

        Parameters
        ----------
        grayord_train : (n_grayord, T_train) float32
            CIFTI cortical grayordinate data for the training set,
            z-scored per run (output of split_train_test()).
        bm_axis       : nibabel BrainModelAxis — from the preprocessed CIFTI header.
        """
        log.info("Converting train data: grayord(%s) → sphere(118584, %d) ...",
                 grayord_train.shape, grayord_train.shape[1])
        sphere_train = grayord_to_sphere_space(grayord_train, bm_axis)
        log.info("  sphere_train shape: %s", sphere_train.shape)
        # MssCf.make_dm() expects (n_sphere_verts, T) — no transpose needed.
        self.make_dm(np.nan_to_num(sphere_train))
        log.info("  Design matrix dm: %s", self.dm.shape)

    # ------------------------------------------------------------------
    # Fit on grayordinate targets
    # ------------------------------------------------------------------

    def fit_grayord(self, grayord_train: np.ndarray) -> None:
        """Fit the banded ridge model with grayordinate targets.

        Parameters
        ----------
        grayord_train : (n_grayord, T_train) float32 — same data used in
                        make_dm_grayord(); targets are the transpose.
        """
        Y_train = grayord_train.T.astype(np.float32)  # (T_train, n_grayord)
        log.info("Fitting model: dm%s → Y%s ...", self.dm.shape, Y_train.shape)
        self.fit(np.nan_to_num(Y_train))
        log.info("  Fit complete.")

    # ------------------------------------------------------------------
    # Test-set evaluation from grayordinate data
    # ------------------------------------------------------------------

    def test_xval_grayord(
        self,
        grayord_test: np.ndarray,
        bm_axis,
    ) -> None:
        """Evaluate the model on the test set using grayordinate data.

        Converts grayord_test to sphere space for building the test design
        matrix (so that make_roi_data indexes the same sphere vertices as
        during training), then evaluates against grayordinate targets.

        After this call:
          self.xval_score       : (n_grayord,) — full-model R² on test set
          self.test_split_scores: [(n_grayord,), (n_grayord,)] — per-band R²

        Parameters
        ----------
        grayord_test : (n_grayord, T_test_concat) float32
            Concatenated test split from all runs (output of split_train_test()),
            z-scored per run independently.
        bm_axis      : nibabel BrainModelAxis.
        """
        log.info("Converting test data: grayord(%s) → sphere(118584, %d) ...",
                 grayord_test.shape, grayord_test.shape[1])
        sphere_test = grayord_to_sphere_space(grayord_test, bm_axis)

        Y_test = grayord_test.T.astype(np.float32)  # (T_test, n_grayord)
        log.info("  Test cross-validation: surf%s → Y%s ...",
                 sphere_test.shape, Y_test.shape)

        # MssCf.test_xval(surf_data, data) expects:
        #   surf_data : (n_sphere_verts, T_test) — for building test DM
        #   data      : (T_test, n_targets)      — for scoring
        self.test_xval(
            np.nan_to_num(sphere_test),
            np.nan_to_num(Y_test),
        )
        log.info(
            "  xval_score: mean=%.4f   test_split_scores: [%.4f, %.4f]",
            np.nanmean(self.xval_score),
            np.nanmean(self.test_split_scores[0]),
            np.nanmean(self.test_split_scores[1]),
        )

    # ------------------------------------------------------------------
    # OLS null model (matches vicsompy test_null_model exactly)
    # ------------------------------------------------------------------

    def compute_null_r2(
        self,
        grayord_test: np.ndarray,
        bm_axis,
    ) -> list:
        """Compute OLS null-model R² for each ROI.

        For each ROI, the null regressor is the mean timecourse across all
        ROI vertices on the test set (same as vicsompy's test_null_model).

        Parameters
        ----------
        grayord_test : (n_grayord, T_test) float32 — test split data.
        bm_axis      : nibabel BrainModelAxis — for sphere-space conversion.

        Returns
        -------
        null_r2 : list of (n_grayord,) float32
            null_r2[i] is the null R² for subsurface i (ROI A or B).
        """
        sphere_test = grayord_to_sphere_space(grayord_test, bm_axis)
        Y_test = grayord_test.T.astype(np.float32)  # (T_test, n_grayord)

        null_r2 = []
        for c, sub in enumerate(self.subsurfaces):
            # subsurface_verts = concat([verts_L, verts_R]) in sphere space
            # (verts_R already has the +59292 offset from generate())
            roi_mean = np.nanmean(sphere_test[sub.subsurface_verts, :], axis=0)
            r2_null = _fit_null_r2(roi_mean, Y_test)
            null_r2.append(r2_null)
            log.info(
                "  Null model [%s]: mean=%.4f  frac>0=%.1f%%",
                self.modalities[c],
                float(np.nanmean(r2_null)),
                100.0 * float(np.mean(r2_null > 0)),
            )
        return null_r2

    # ------------------------------------------------------------------
    # Save all output maps as npy + CIFTI
    # ------------------------------------------------------------------

    def save_all_maps(
        self,
        null_r2: list,
        roi_a: str,
        roi_b: str,
        out_dir: str,
        n_cortex: int = None,
    ) -> None:
        """Save all held-out CF R² maps as clearly named ``.npy`` files.

        CIFTI output is handled downstream by integration_maps.py, which
        writes a single combined dscalar with all scalar and derived maps.

        Parameters
        ----------
        null_r2   : output of compute_null_r2() — list of (n_grayord,) float32.
        roi_a, roi_b : str — ROI names (used in file names).
        out_dir   : str — directory for npy output.
        n_cortex  : int or None — when set, slice output arrays to the first
                    n_cortex grayordinates (used when fitting on full-brain data
                    to strip subcortex before saving cortex-only maps).
        """
        os.makedirs(out_dir, exist_ok=True)

        # ── Unpack results from test_xval_grayord and get_params ────────────
        R2_full    = np.asarray(self.xval_score,           dtype=np.float32)
        R2_a       = np.asarray(self.test_split_scores[0], dtype=np.float32)
        R2_b       = np.asarray(self.test_split_scores[1], dtype=np.float32)
        R2_null_a  = null_r2[0].astype(np.float32)
        R2_null_b  = null_r2[1].astype(np.float32)

        # Slice to cortex-only when full-brain fitting was used
        if n_cortex is not None:
            R2_full   = R2_full[:n_cortex]
            R2_a      = R2_a[:n_cortex]
            R2_b      = R2_b[:n_cortex]
            R2_null_a = R2_null_a[:n_cortex]
            R2_null_b = R2_null_b[:n_cortex]
            log.info("  Sliced to cortex-only: %d grayords", n_cortex)

        R2_a_nc = (R2_a - R2_null_a).astype(np.float32)
        R2_b_nc = (R2_b - R2_null_b).astype(np.float32)

        # product_map: geometric mean of raw R² — measures joint prediction
        # independent of the null-model baseline.
        product_map = np.sqrt(
            np.clip(R2_a, 0, None) * np.clip(R2_b, 0, None)
        ).astype(np.float32)

        # product_map_nc: geometric mean of null-corrected R² — measures joint
        # topographic response above the global-signal baseline.
        product_map_nc = np.sqrt(
            np.clip(R2_a_nc, 0, None) * np.clip(R2_b_nc, 0, None)
        ).astype(np.float32)

        stems = cf_model_map_stems(roi_a, roi_b)
        maps = {
            stems["full_r2"]: R2_full,
            stems["split_r2_a"]: R2_a,
            stems["split_r2_b"]: R2_b,
            stems["roi_mean_null_r2_a"]: R2_null_a,
            stems["roi_mean_null_r2_b"]: R2_null_b,
            stems["null_corrected_split_r2_a"]: R2_a_nc,
            stems["null_corrected_split_r2_b"]: R2_b_nc,
            stems["joint_split_r2_geomean"]: product_map,
            stems["joint_null_corrected_split_r2_geomean"]: product_map_nc,
        }

        # ── Log summary stats ───────────────────────────────────────────────
        log.info("Map summary:")
        for name, arr in maps.items():
            log.info(
                "  %-25s mean=%+.4f  frac>0=%.1f%%",
                name, float(np.nanmean(arr)), 100.0 * float(np.mean(arr > 0)),
            )

        # ── Save npy ────────────────────────────────────────────────────────
        for name, arr in maps.items():
            npy_path = os.path.join(out_dir, f"{name}.npy")
            np.save(npy_path, arr)

        # Also save band_sizes so downstream scripts can read band structure.
        band_sizes = np.array([2 * n for n in self.comps_per_hem], dtype=np.int32)
        np.save(os.path.join(out_dir, "band_sizes.npy"), band_sizes)
        log.info("  band_sizes: %s", band_sizes.tolist())

        log.info("npy maps saved to: %s", out_dir)

    # ------------------------------------------------------------------
    # GPU → CPU offload before split-predict (OOM prevention)
    # ------------------------------------------------------------------

    def _offload_fitted_to_cpu(self) -> None:
        """Move fitted GPU tensors to CPU and switch himalaya backend to torch.

        Called once before get_params() and test_xval() because those steps
        call pipeline.predict(split=True) which allocates
        (n_kernels × T × n_targets) on GPU — too large for 12 GB VRAM.
        Fitting (the expensive random-search part) was already done on GPU;
        scoring on CPU costs only a few seconds.
        """
        import gc
        try:
            import torch
            from himalaya.backend import set_backend

            if not torch.cuda.is_available():
                return

            def _to_cpu(obj, _seen=None):
                """Recursively move CUDA tensors to CPU in any sklearn/himalaya object."""
                if _seen is None:
                    _seen = set()
                oid = id(obj)
                if oid in _seen:
                    return
                _seen.add(oid)
                for attr, val in list(getattr(obj, '__dict__', {}).items()):
                    if isinstance(val, torch.Tensor) and val.is_cuda:
                        setattr(obj, attr, val.cpu())
                    elif isinstance(val, (list, tuple)):
                        moved = [
                            item.cpu() if isinstance(item, torch.Tensor) and item.is_cuda
                            else item
                            for item in val
                        ]
                        try:
                            setattr(obj, attr, type(val)(moved))
                        except TypeError:
                            pass
                    elif hasattr(val, '__dict__'):
                        _to_cpu(val, _seen)

            _to_cpu(self.pipeline)
            gc.collect()
            torch.cuda.empty_cache()
            set_backend("torch")
            log.info("  Fitted pipeline → CPU; GPU cache cleared; backend → torch.")
        except Exception as exc:
            log.warning("  _offload_fitted_to_cpu() failed (%s) — proceeding on GPU.", exc)

    def get_params(self) -> None:
        """get_params() with GPU→CPU offload to avoid OOM in split prediction."""
        self._offload_fitted_to_cpu()
        super().get_params()

    def test_xval(self, surf_data: np.ndarray, data: np.ndarray) -> None:
        """test_xval() using the CPU backend set by get_params()/_offload_fitted_to_cpu()."""
        # _offload_fitted_to_cpu() was already called in get_params() so the
        # pipeline state and himalaya backend are already on CPU here.
        super().test_xval(surf_data, data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_subsurfaces_injected(self):
        """Raise if inject_subsurfaces() has not been called."""
        if not hasattr(self, 'subsurfaces') or self.subsurfaces is None:
            raise RuntimeError(
                "Call inject_subsurfaces() before make_dm_grayord() or test_xval_grayord()."
            )

