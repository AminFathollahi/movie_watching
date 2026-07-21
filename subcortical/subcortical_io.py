"""
subcortical/subcortical_io.py
==============================
Geometry plumbing for the subcortical RSA/encoding extension: extraction,
per-structure column bookkeeping, k-NN neighbor arrays (geodesic + Euclidean
fallback), nucleus coordinate masks, and the CIFTI template used to save
subcortical maps.

Everything downstream (subcortical_rsa.py, subcortical_encoding.py) reuses
rsa/shared/rsa_utils.py, rsa/searchlight.py::run_searchlight, and cifti_io.py
unchanged. This file is the only new geometry logic (see subcortex.txt).
"""

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocess_individual import RUN_IDS, get_run_path, preprocess_run, save_dtseries  # noqa: E402
from cifti_io import save_cifti_map  # noqa: E402 (unused here but re-exported for callers)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Structure catalogue
# =============================================================================

SUBCORTICAL_STRUCTURES = [
    "CEREBELLUM_LEFT", "CEREBELLUM_RIGHT", "BRAIN_STEM",
    "THALAMUS_LEFT", "THALAMUS_RIGHT", "CAUDATE_LEFT", "CAUDATE_RIGHT",
    "PUTAMEN_LEFT", "PUTAMEN_RIGHT", "PALLIDUM_LEFT", "PALLIDUM_RIGHT",
    "ACCUMBENS_LEFT", "ACCUMBENS_RIGHT", "AMYGDALA_LEFT", "AMYGDALA_RIGHT",
    "HIPPOCAMPUS_LEFT", "HIPPOCAMPUS_RIGHT",
    "DIENCEPHALON_VENTRAL_LEFT", "DIENCEPHALON_VENTRAL_RIGHT",
]

# Neighbor-mode policy (user directive): geodesic everywhere; Euclidean is
# only the automatic fallback inside build_neighbors when a structure's
# in-mask voxel graph is disconnected. Cerebellum additionally gets the
# purist SUIT surface geodesic (build_cerebellum_surface_neighbors).
STRUCTURE_NEIGHBOR_MODE = {s: "geodesic" for s in SUBCORTICAL_STRUCTURES}

K_DEFAULT = 100

# ponytail: coord spheres (peaks from Sitek 2019, Front. Neurosci.; standard
# geniculate coordinates). Upgrade to Sitek/Bianciardi probabilistic atlas
# masks here if the user wants true anatomical masks instead of spheres.
NUCLEI_SEEDS = {
    "IC_L": ((-6, -33, -11), "BRAIN_STEM"),      "IC_R": ((6, -33, -11), "BRAIN_STEM"),
    "SC_L": ((-4, -31, -4),  "BRAIN_STEM"),      "SC_R": ((4, -31, -4),  "BRAIN_STEM"),
    "MGN_L": ((-17, -24, -2), "THALAMUS_LEFT"),  "MGN_R": ((17, -24, -2), "THALAMUS_RIGHT"),
    "LGN_L": ((-23, -24, -3), "THALAMUS_LEFT"),  "LGN_R": ((23, -24, -3), "THALAMUS_RIGHT"),
}


# =============================================================================
# Extraction
# =============================================================================

def extract_subcortical(img, structures=SUBCORTICAL_STRUCTURES):
    """Extract the requested subcortical structures from a CIFTI dtseries image.

    Mirrors preprocess_individual.extract_cortex, but selects arbitrary named
    structures and reorders columns to match `structures` (not CIFTI's
    on-disk order), so downstream column ranges are stable regardless of
    file layout.

    Returns
    -------
    data    : (n_vox, T) float32
    bm_axis : sliced BrainModelAxis, columns in `structures` order
    """
    bm_axis = img.header.get_axis(1)
    full_data = img.get_fdata(dtype=np.float32)  # (T, n_all)

    by_name = {}
    for name, sl, _ in bm_axis.iter_structures():
        by_name[name] = np.arange(len(bm_axis))[sl]

    cols = []
    for s in structures:
        key = f"CIFTI_STRUCTURE_{s}"
        if key not in by_name:
            raise RuntimeError(f"Structure {key} not found in BrainModelAxis.")
        cols.append(by_name[key])
    cols = np.concatenate(cols)

    return full_data[:, cols].T, bm_axis[cols]


def preprocess_subject_subcortical(sub: str, raw_dir: Path, tr: float, args,
                                    structures=SUBCORTICAL_STRUCTURES) -> tuple:
    """Subcortical sibling of preprocess_individual.preprocess_subject.

    Reuses get_run_path + preprocess_run verbatim; only the extraction step
    differs (extract_subcortical instead of extract_cortex).
    """
    bm_axis = None
    run_chunks = []
    run_trs = []

    for run_id in RUN_IDS:
        path = get_run_path(Path(raw_dir), sub, run_id)
        img = nib.load(str(path))
        data, bm_ax = extract_subcortical(img, structures)
        if bm_axis is None:
            bm_axis = bm_ax
        run_trs.append(img.shape[0])

        preprocess_run(data, args)
        run_chunks.append(data)

    combined = np.concatenate(run_chunks, axis=1)
    return combined, bm_axis, np.array(run_trs, dtype=np.int32)


# =============================================================================
# Group-average timeseries (mirrors preprocess_individual's running-mean pattern)
# =============================================================================

def compute_group_average_subcortical(subjects: list, raw_dir: Path, tr: float, args,
                                       structures=SUBCORTICAL_STRUCTURES,
                                       out_path: Path = None) -> tuple:
    """Running-mean group-average subcortical timeseries across subjects.

    Same accumulator pattern as preprocess_individual.py's --save-average,
    calling preprocess_subject_subcortical instead of the cortical loader.
    Saves via preprocess_individual.save_dtseries (bm_axis-agnostic) if
    out_path is given.

    Returns (group_mean (n_vox, T) float32, bm_axis, run_trs).
    """
    running_mean = None
    bm_axis_ref = None
    run_trs_ref = None
    n_avg = 0
    skipped = []

    for idx, sub in enumerate(subjects, 1):
        log.info(f"  [{idx:03d}/{len(subjects):03d}] {sub} ...")
        try:
            data, bm_axis, run_trs = preprocess_subject_subcortical(
                sub, raw_dir, tr, args, structures)
        except Exception as exc:
            log.warning(f"    SKIPPED {sub}: {exc}")
            skipped.append(sub)
            continue

        if running_mean is None:
            running_mean = np.zeros_like(data, dtype=np.float64)
            bm_axis_ref = bm_axis
            run_trs_ref = run_trs

        if data.shape != running_mean.shape:
            log.warning(f"    SKIPPED {sub}: shape {data.shape} != {running_mean.shape}")
            skipped.append(sub)
            continue

        n_avg += 1
        running_mean += (data - running_mean) / n_avg

    if skipped:
        log.warning(f"  Skipped {len(skipped)}/{len(subjects)}: {skipped}")
    log.info(f"  Averaged {n_avg}/{len(subjects)} subjects.")

    group_mean = running_mean.astype(np.float32)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        save_dtseries(group_mean, bm_axis_ref, Path(out_path), tr)
        trs_path = Path(str(out_path).replace("_subcortical.dtseries.nii", "_run_trs.npy"))
        np.save(str(trs_path), run_trs_ref)
        log.info(f"  Saved: {out_path}  shape={group_mean.shape}  run_trs={run_trs_ref.tolist()}")

    return group_mean, bm_axis_ref, run_trs_ref


# =============================================================================
# Column bookkeeping
# =============================================================================

def struct_slices(bm_axis) -> dict:
    """Per-structure column ranges + MNI world coords.

    Subcortical analog of cifti_io.get_cortex_vertex_indices. Requires
    bm_axis to already be sliced/ordered by extract_subcortical (so each
    structure's columns are contiguous).
    """
    out = {}
    for name, sl, model in bm_axis.iter_structures():
        short = name.replace("CIFTI_STRUCTURE_", "")
        idx = np.arange(len(bm_axis))[sl]
        ijk = model.voxel
        world = nib.affines.apply_affine(bm_axis.affine, ijk)
        out[short] = {
            "start": int(idx[0]), "stop": int(idx[-1]) + 1,
            "voxel_ijk": ijk, "world_xyz": world,
        }
    return out


# =============================================================================
# Neighbor arrays (k-NN within one structure)
# =============================================================================

def _knn_euclidean(world_xyz: np.ndarray, k: int) -> np.ndarray:
    n = len(world_xyz)
    kk = min(k, n)
    tree = cKDTree(world_xyz)
    _, idx = tree.query(world_xyz, k=kk)
    if kk == 1:
        idx = idx[:, None]
    return idx.astype(np.int32)


def _knn_geodesic(voxel_ijk: np.ndarray, world_xyz: np.ndarray, k: int,
                   batch: int = 1000) -> np.ndarray:
    """Within-mask voxel-graph shortest path k-NN (26-connectivity, mm-weighted).

    Falls back per-source to Euclidean-nearest when fewer than k nodes are
    graph-reachable (disconnected component) — rare for these compact masks.
    """
    n = len(voxel_ijk)
    kk = min(k, n)

    tree_vox = cKDTree(voxel_ijk.astype(np.float64))
    pairs = tree_vox.query_pairs(r=np.sqrt(3) + 1e-6, output_type="ndarray")
    if len(pairs) == 0:
        log.warning("  No 26-connected edges found — falling back to Euclidean k-NN entirely.")
        return _knn_euclidean(world_xyz, k)

    i, j = pairs[:, 0], pairs[:, 1]
    w = np.linalg.norm(world_xyz[i] - world_xyz[j], axis=1)
    graph = coo_matrix(
        (np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n),
    ).tocsr()

    neighbors = np.empty((n, kk), dtype=np.int32)
    euclid_neighbors = None  # lazily computed only if a disconnected source needs it

    for start in range(0, n, batch):
        end = min(start + batch, n)
        idx_batch = np.arange(start, end)
        dist = dijkstra(graph, indices=idx_batch, directed=False)  # (batch, n)
        for local_i, global_i in enumerate(idx_batch):
            row = dist[local_i]
            reached = np.isfinite(row)
            n_reached = int(reached.sum())
            if n_reached >= kk:
                part = np.argpartition(row, kk - 1)[:kk]
                order = part[np.argsort(row[part])]
            else:
                reached_idx = np.where(reached)[0]
                order = reached_idx[np.argsort(row[reached_idx])]
                if euclid_neighbors is None:
                    euclid_neighbors = _knn_euclidean(world_xyz, n)  # full ranking
                already = set(order.tolist())
                fill = [x for x in euclid_neighbors[global_i] if x not in already]
                order = np.concatenate([order, np.array(fill[: kk - len(order)], dtype=np.int32)])
                if len(order) < kk:
                    order = np.concatenate(
                        [order, np.full(kk - len(order), global_i, dtype=np.int32)])
            neighbors[global_i] = order[:kk]

    return neighbors


def build_neighbors(voxel_ijk: np.ndarray, world_xyz: np.ndarray, k: int,
                     mode: str, cache_path: Path = None) -> np.ndarray:
    """(n_vox, k) int32 local k-NN indices within one structure.

    Cached to `cache_path` (subject-invariant grid → compute once). Padded
    to width k by repeating each vertex's own index if n_vox < k.
    """
    if cache_path is not None and Path(cache_path).exists():
        return np.load(str(cache_path))

    n = len(world_xyz)
    if mode == "euclidean":
        neighbors = _knn_euclidean(world_xyz, k)
    elif mode == "geodesic":
        neighbors = _knn_geodesic(voxel_ijk, world_xyz, k)
    elif mode == "surface_geodesic":
        raise ValueError("surface_geodesic is only available for cerebellum via "
                          "build_cerebellum_surface_neighbors().")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if n < k:
        pad_width = k - neighbors.shape[1]
        self_col = np.arange(n, dtype=np.int32)[:, None]
        pad = np.tile(self_col, (1, pad_width))
        neighbors = np.concatenate([neighbors, pad], axis=1)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), neighbors)
    return neighbors


# =============================================================================
# Purist cerebellar SURFACE geodesic (SUIT flatmap surface)
# =============================================================================
# ponytail: purist = SUIT surface geodesic (required by user directive).
# Voxel-graph geodesic (build_neighbors mode="geodesic") is the guaranteed
# fallback if SUITPy/wb_command are unavailable at run time; Euclidean is
# not an acceptable fallback for cerebellum.

def build_cerebellum_surface_neighbors(world_xyz: np.ndarray, k: int,
                                        cache_dir: Path, workbench: str) -> np.ndarray:
    """(n_vox, k) int32 k-NN by SUIT surface geodesic distance.

    Maps each cerebellar CIFTI voxel to its nearest SUIT PIAL_FSL (MNI-space)
    surface vertex, computes the all-to-all geodesic distance on the
    anatomically folded PIAL_SUIT mesh (via the same wb_command wrapper the
    cortical pipeline uses), then takes each voxel's k nearest neighbors by
    surface-geodesic distance (through its mapped vertex).

    Raises on any failure (missing SUITPy, missing wb_command, etc.) — the
    caller is responsible for catching and falling back to voxel-graph
    geodesic, per the required-fallback policy above.
    """
    import SUITPy.flatmap as flatmap
    sys.path.insert(0, str(ROOT / "rsa"))
    from searchlight import _compute_geodesic_dconn  # noqa: E402

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    neighbors_path = cache_dir / f"cerebellum_neighbors_k{k}_surface_geodesic.npy"
    vox2vert_path = cache_dir / "cerebellum_vox2vert.npy"

    surf_dir = Path(flatmap._base_dir) / "surfaces"
    pial_fsl = surf_dir / "PIAL_FSL.surf.gii"     # SUIT mesh, MNI-space coordinates (for mapping)
    pial_suit = surf_dir / "PIAL_SUIT.surf.gii"   # same mesh, anatomically folded (for geodesic)

    surf_verts_mni = nib.load(str(pial_fsl)).darrays[0].data.astype(np.float64)  # (n_suit, 3)
    n_suit = surf_verts_mni.shape[0]

    if vox2vert_path.exists():
        vox2vert = np.load(str(vox2vert_path))
    else:
        tree = cKDTree(surf_verts_mni)
        _, vox2vert = tree.query(world_xyz)
        np.save(str(vox2vert_path), vox2vert)
    log.info(f"  [SUIT] mapped {len(world_xyz)} cerebellar voxels onto {n_suit} SUIT surface vertices")

    dconn_path = cache_dir / "suit_cerebellum_geodesic.dconn.nii"
    _compute_geodesic_dconn(str(pial_suit), workbench, dconn_path)

    img = nib.load(str(dconn_path))
    full = np.asarray(img.dataobj, dtype=np.float32)  # (n_suit, n_suit) ~3.3 GB
    rows = full[vox2vert]          # (n_vox, n_suit)
    dist_vv = rows[:, vox2vert]    # (n_vox, n_vox)
    del full, rows
    np.fill_diagonal(dist_vv, 0.0)

    n_vox = len(world_xyz)
    kk = min(k, n_vox)
    order = np.argsort(dist_vv, axis=1)[:, :kk].astype(np.int32)
    if n_vox < k:
        pad = np.tile(np.arange(n_vox, dtype=np.int32)[:, None], (1, k - n_vox))
        order = np.concatenate([order, pad], axis=1)

    np.save(str(neighbors_path), order)
    return order


def get_cerebellum_neighbors(hem: str, world_xyz: np.ndarray, voxel_ijk: np.ndarray,
                              k: int, cache_dir: Path, workbench: str) -> tuple[np.ndarray, str]:
    """Try purist SUIT surface geodesic for cerebellum; fall back to voxel-graph geodesic.

    Returns (neighbors, mode_used).
    """
    hem_cache_dir = Path(cache_dir) / f"CEREBELLUM_{hem}"
    try:
        neighbors = build_cerebellum_surface_neighbors(world_xyz, k, hem_cache_dir, workbench)
        return neighbors, "surface_geodesic"
    except Exception as exc:
        log.warning(f"  [SUIT] surface geodesic unavailable for CEREBELLUM_{hem} "
                    f"({type(exc).__name__}: {exc}) — falling back to voxel-graph geodesic.")
        cache_path = Path(cache_dir) / f"CEREBELLUM_{hem}_neighbors_k{k}_geodesic.npy"
        neighbors = build_neighbors(voxel_ijk, world_xyz, k, "geodesic", cache_path)
        return neighbors, "geodesic"


# =============================================================================
# Nucleus coordinate masks
# =============================================================================

def build_nucleus_masks(struct_slices_dict: dict, radius: float = 5.0) -> dict:
    """MNI coordinate spheres intersected with each nucleus's parent structure.

    Returns dict[nucleus -> (n_nuc,) int array of GLOBAL column indices into
    the subcortical data matrix].
    """
    masks = {}
    for nucleus, (seed, parent) in NUCLEI_SEEDS.items():
        if parent not in struct_slices_dict:
            raise RuntimeError(f"Parent structure {parent} for nucleus {nucleus} not in data.")
        info = struct_slices_dict[parent]
        d = np.linalg.norm(info["world_xyz"] - np.asarray(seed), axis=1)
        local_idx = np.where(d <= radius)[0]
        if local_idx.size == 0:
            raise AssertionError(f"Nucleus mask {nucleus} is empty (seed={seed}, r={radius}).")
        global_idx = info["start"] + local_idx
        masks[nucleus] = global_idx
        log.info(f"  Nucleus {nucleus}: {len(global_idx)} voxels (parent={parent})")
    return masks


# =============================================================================
# CIFTI template
# =============================================================================

def make_subcortical_template(structures: list, out_path: Path,
                               raw_dir: Path, sub: str = "132118") -> Path:
    """Write a 1-timepoint dscalar CIFTI whose BrainModelAxis is exactly the
    concatenated subcortical structures, for use as --template-cifti.
    """
    img = nib.load(str(get_run_path(Path(raw_dir), sub, RUN_IDS[0])))
    _, bm_axis = extract_subcortical(img, structures)
    n_vox = len(bm_axis)

    scalar_axis = nib.cifti2.ScalarAxis(["template"])
    header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, bm_axis))
    out_img = nib.Cifti2Image(np.zeros((1, n_vox), dtype=np.float32), header=header)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))
    log.info(f"  Template saved: {out_path}  n_vox={n_vox}")
    return Path(out_path)


# =============================================================================
# Self-check
# =============================================================================

def demo():
    RAW_DIR = Path("/home/amin/Research/Representation/Movie/data/individual-59k")
    SUB = "132118"
    CACHE_DIR = ROOT / "outputs_subcortical_selfcheck_cache"

    print("== extract_subcortical ==")
    img = nib.load(str(get_run_path(RAW_DIR, SUB, RUN_IDS[0])))
    data, bm_axis = extract_subcortical(img, SUBCORTICAL_STRUCTURES)
    expected_n = sum(1 for name, _, _ in bm_axis.iter_structures() for _ in [0])
    assert data.shape[0] == len(bm_axis), "n_vox mismatch between data and bm_axis"
    assert data.shape == (len(bm_axis), img.shape[0]), f"bad data shape {data.shape}"
    print(f"  n_vox={data.shape[0]}  T={data.shape[1]}  OK")

    print("== struct_slices ==")
    slices = struct_slices(bm_axis)
    assert set(slices.keys()) == set(SUBCORTICAL_STRUCTURES), "structure name mismatch"
    order = sorted(slices.values(), key=lambda d: d["start"])
    assert order[0]["start"] == 0, "first slice must start at 0"
    assert order[-1]["stop"] == len(bm_axis), "last slice must cover full data"
    for a, b in zip(order, order[1:]):
        assert a["stop"] == b["start"], "slices are not contiguous/non-overlapping"
    print(f"  {len(slices)} structures, contiguous & covering all {len(bm_axis)} columns  OK")

    print("== build_neighbors (cerebellum voxel-graph geodesic vs Euclidean) ==")
    cb = slices["CEREBELLUM_LEFT"]
    k = 100
    neigh_geo = build_neighbors(cb["voxel_ijk"], cb["world_xyz"], k, "geodesic")
    neigh_euc = build_neighbors(cb["voxel_ijk"], cb["world_xyz"], k, "euclidean")
    assert neigh_geo.shape == (cb["stop"] - cb["start"], k)
    assert neigh_geo.min() >= 0 and neigh_geo.max() < cb["stop"] - cb["start"]
    assert np.all(neigh_geo[:, 0] == np.arange(neigh_geo.shape[0])), "row 0 must be self"
    assert np.all(neigh_euc[:, 0] == np.arange(neigh_euc.shape[0])), "row 0 must be self"
    n_diff = np.mean([set(neigh_geo[i].tolist()) != set(neigh_euc[i].tolist())
                       for i in range(neigh_geo.shape[0])])
    assert n_diff > 0.05, "geodesic and Euclidean neighbor sets are suspiciously identical"
    print(f"  shape={neigh_geo.shape}  self-first OK  {n_diff*100:.1f}% of rows differ from Euclidean  OK")

    print("== neighbor caches are subject-invariant ==")
    img2 = nib.load(str(get_run_path(RAW_DIR, "100610", RUN_IDS[0])))
    _, bm_axis2 = extract_subcortical(img2, SUBCORTICAL_STRUCTURES)
    slices2 = struct_slices(bm_axis2)
    cb2 = slices2["CEREBELLUM_LEFT"]
    assert np.array_equal(cb["voxel_ijk"], cb2["voxel_ijk"]), "voxel grid differs across subjects!"
    assert np.allclose(cb["world_xyz"], cb2["world_xyz"]), "world coords differ across subjects!"
    print("  132118 and 100610 share identical cerebellum voxel grid  OK")

    print("== build_nucleus_masks ==")
    masks = build_nucleus_masks(slices, radius=5.0)
    for nucleus, (seed, parent) in NUCLEI_SEEDS.items():
        idx = masks[nucleus]
        assert idx.size > 0, f"{nucleus} mask empty"
        pinfo = slices[parent]
        assert idx.min() >= pinfo["start"] and idx.max() < pinfo["stop"], \
            f"{nucleus} indices fall outside parent {parent}"
        print(f"  {nucleus}: {idx.size} vox (parent={parent})  OK")

    print("== make_subcortical_template ==")
    tmpl_path = CACHE_DIR / "subcortical_template.dscalar.nii"
    make_subcortical_template(SUBCORTICAL_STRUCTURES, tmpl_path, RAW_DIR, SUB)
    reloaded = nib.load(str(tmpl_path))
    assert reloaded.header.get_axis(1).__len__() if hasattr(reloaded.header.get_axis(1), "__len__") \
        else len(reloaded.header.get_axis(1)) == len(bm_axis)
    assert len(reloaded.header.get_axis(1)) == len(bm_axis), "template bm_axis length mismatch"
    print(f"  template reloads with {len(reloaded.header.get_axis(1))} columns  OK")

    print("\nALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
