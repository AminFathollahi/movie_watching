#!/usr/bin/env python3
"""Build the Workbench bundle for subcortical RSA results.

The analysis grid is a concatenated CIFTI BrainModelAxis because that is the
most convenient internal representation for the shared RSA kernels.  It is
not, however, the public visualization format.  This module converts every
group-level result into independent anatomical units:

* one SUIT cerebellum metric;
* one brain-stem metric; and
* one bilateral metric for every paired subcortical structure.

The generated spec intentionally contains no dense-scalar CIFTI and no
all-subcortical/whole-hemisphere mesh.  Each metric is attached only to its
matching structure mesh, and every exported filename fully encodes the
analysis that produced it (subject scope, model/modality, config, result
name, structure) so files can be told apart and loaded by hand.

This is the only module that writes into workbench_visualization/ -- it owns
mesh generation, SUIT/atlas asset staging, and metric export end to end.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from subcortical_io import struct_slices  # noqa: E402
from paths import OUTPUTS  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = OUTPUTS / "subcortical"
DEFAULT_WB = Path("/opt/workbench/bin_linux64/wb_command")

# ``carrier`` is Workbench's fixed structure enum.  For a bilateral mesh we
# use that anatomical area's LEFT enum as a unique carrier.  This avoids the
# vertex-count collisions caused by assigning many differently sized meshes
# to OTHER, while the filename/manifest explicitly record that the geometry
# and data are bilateral.
ANATOMICAL_GROUPS = {
    "CEREBELLUM": {
        "parts": ("CEREBELLUM_LEFT", "CEREBELLUM_RIGHT"),
        "carrier": "CEREBELLUM",
        "surface": "suit",
    },
    "BRAIN_STEM": {
        "parts": ("BRAIN_STEM",),
        "carrier": "BRAIN_STEM",
        "surface": "boundary",
    },
    "BILATERAL_THALAMUS": {
        "parts": ("THALAMUS_LEFT", "THALAMUS_RIGHT"),
        "carrier": "THALAMUS_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_CAUDATE": {
        "parts": ("CAUDATE_LEFT", "CAUDATE_RIGHT"),
        "carrier": "CAUDATE_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_PUTAMEN": {
        "parts": ("PUTAMEN_LEFT", "PUTAMEN_RIGHT"),
        "carrier": "PUTAMEN_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_PALLIDUM": {
        "parts": ("PALLIDUM_LEFT", "PALLIDUM_RIGHT"),
        "carrier": "PALLIDUM_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_ACCUMBENS": {
        "parts": ("ACCUMBENS_LEFT", "ACCUMBENS_RIGHT"),
        "carrier": "ACCUMBENS_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_AMYGDALA": {
        "parts": ("AMYGDALA_LEFT", "AMYGDALA_RIGHT"),
        "carrier": "AMYGDALA_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_HIPPOCAMPUS": {
        "parts": ("HIPPOCAMPUS_LEFT", "HIPPOCAMPUS_RIGHT"),
        "carrier": "HIPPOCAMPUS_LEFT",
        "surface": "boundary",
    },
    "BILATERAL_VENTRAL_DIENCEPHALON": {
        "parts": ("DIENCEPHALON_VENTRAL_LEFT", "DIENCEPHALON_VENTRAL_RIGHT"),
        "carrier": "DIENCEPHALON_VENTRAL_LEFT",
        "surface": "boundary",
    },
}

CEREBELLAR_ATLASES = {
    "Diedrichsen_2009": {"maps": ["atl-Anatom"], "space": "MNI"},
    "Buckner_2011": {"maps": ["atl-Buckner7", "atl-Buckner17"], "space": "MNI"},
    "King_2019": {"maps": ["atl-MDTB10"], "space": "SUIT"},
    "Ji_2019": {"maps": ["atl-Ji10"], "space": "MNI"},
}


# =============================================================================
# Low-level surface / atlas helpers
# =============================================================================

def save_surface(path: Path, coords: np.ndarray, faces: np.ndarray) -> None:
    img = GiftiImage(darrays=[
        GiftiDataArray(coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
        GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def load_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return (
        img.darrays[0].data.astype(np.float32),
        img.darrays[1].data.astype(np.int32),
    )


def boundary_mesh(voxel_ijk: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a block boundary mesh and vertex-to-voxel incidence matrix.

    Intentionally dependency-light: each exposed voxel face contributes two
    triangles; shared voxel-corner vertices are merged.
    """
    vox = voxel_ijk.astype(np.int32)
    vox_set = {tuple(v) for v in vox.tolist()}

    face_defs = [
        ((-1, 0, 0), [(-.5, -.5, -.5), (-.5,  .5, -.5), (-.5,  .5,  .5), (-.5, -.5,  .5)]),
        (( 1, 0, 0), [( .5, -.5, -.5), ( .5, -.5,  .5), ( .5,  .5,  .5), ( .5,  .5, -.5)]),
        ((0, -1, 0), [(-.5, -.5, -.5), (-.5, -.5,  .5), ( .5, -.5,  .5), ( .5, -.5, -.5)]),
        ((0,  1, 0), [(-.5,  .5, -.5), ( .5,  .5, -.5), ( .5,  .5,  .5), (-.5,  .5,  .5)]),
        ((0, 0, -1), [(-.5, -.5, -.5), ( .5, -.5, -.5), ( .5,  .5, -.5), (-.5,  .5, -.5)]),
        ((0, 0,  1), [(-.5, -.5,  .5), (-.5,  .5,  .5), ( .5,  .5,  .5), ( .5, -.5,  .5)]),
    ]

    vert_index: dict[tuple[float, float, float], int] = {}
    verts_ijk: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_voxels: list[list[int]] = []

    def add_vertex(corner: tuple[float, float, float], voxel_i: int) -> int:
        key = (round(corner[0], 3), round(corner[1], 3), round(corner[2], 3))
        idx = vert_index.get(key)
        if idx is None:
            idx = len(verts_ijk)
            vert_index[key] = idx
            verts_ijk.append(key)
            vertex_voxels.append([voxel_i])
        else:
            vertex_voxels[idx].append(voxel_i)
        return idx

    for voxel_i, v in enumerate(vox):
        base = tuple(int(x) for x in v)
        for direction, offsets in face_defs:
            neighbor = (base[0] + direction[0], base[1] + direction[1], base[2] + direction[2])
            if neighbor in vox_set:
                continue
            q = []
            for off in offsets:
                corner = (base[0] + off[0], base[1] + off[1], base[2] + off[2])
                q.append(add_vertex(corner, voxel_i))
            faces.append((q[0], q[1], q[2]))
            faces.append((q[0], q[2], q[3]))

    ijk = np.asarray(verts_ijk, dtype=np.float64)
    coords = nib.affines.apply_affine(affine, ijk)
    return coords, np.asarray(faces, dtype=np.int32), np.asarray(vertex_voxels, dtype=object)


def set_structure(wb_command: Path, filename: Path, structure: str,
                  surface_type: str | None = None,
                  secondary_type: str | None = None) -> None:
    cmd = [str(wb_command), "-set-structure", str(filename), structure]
    if surface_type is not None:
        cmd.extend(["-surface-type", surface_type])
    if secondary_type is not None:
        cmd.extend(["-surface-secondary-type", secondary_type])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def copy_suit_assets(out_dir: Path, wb_command: Path) -> dict:
    import SUITPy.flatmap as flatmap

    src = Path(flatmap._base_dir) / "surfaces"
    dst = out_dir / "surfaces" / "suit"
    dst.mkdir(parents=True, exist_ok=True)
    names = [
        "FLAT.surf.gii", "PIAL_FSL.surf.gii", "WHITE_FSL.surf.gii",
        "PIAL_SUIT.surf.gii", "WHITE_SUIT.surf.gii", "SUIT.shape.gii",
        "borders.txt",
    ]
    copied = {}
    for name in names:
        target = dst / name
        if not target.exists():
            shutil.copy2(src / name, target)
        if name.endswith(".surf.gii"):
            if name == "FLAT.surf.gii":
                set_structure(wb_command, target, "CEREBELLUM", surface_type="FLAT")
            elif "PIAL" in name:
                set_structure(wb_command, target, "CEREBELLUM", surface_type="ANATOMICAL", secondary_type="PIAL")
            elif "WHITE" in name:
                set_structure(wb_command, target, "CEREBELLUM", surface_type="ANATOMICAL", secondary_type="GRAY_WHITE")
        elif name.endswith(".shape.gii"):
            set_structure(wb_command, target, "CEREBELLUM")
        copied[name] = str(target)
    return copied


def fetch_cerebellar_parcellations(out_dir: Path, wb_command: Path, suit_assets: dict,
                                   skip_download: bool = False) -> dict:
    atlas_root = out_dir / "atlases" / "cerebellar_flatmap"
    atlas_root.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        import SUITPy as suit

        for atlas, cfg in CEREBELLAR_ATLASES.items():
            try:
                suit.fetch_atlas(
                    atlas,
                    maps=cfg["maps"],
                    space=cfg["space"],
                    atlas_dir=str(atlas_root),
                    verbose=1,
                )
            except Exception as exc:
                log.warning("Could not fetch %s: %s: %s", atlas, type(exc).__name__, exc)

    flat = Path(suit_assets["FLAT.surf.gii"])
    fetched = {}
    for label_path in sorted(atlas_root.glob("*/*.label.gii")):
        name = label_path.stem.replace("_dseg.label", "")
        set_structure(wb_command, label_path, "CEREBELLUM")
        border_path = label_path.with_suffix("").with_suffix(".border")
        subprocess.run(
            [str(wb_command), "-label-to-border", str(flat), str(label_path), str(border_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        fetched[name] = {"label": str(label_path), "border": str(border_path)}
    return fetched


def map_cerebellum_to_suit(vol_img: nib.Nifti1Image, suit_assets: dict) -> np.ndarray:
    import SUITPy.flatmap as flatmap

    data = flatmap.vol_to_surf(
        vol_img,
        space="FSL",
        ignore_zeros=False,
        stats="nanmean",
        inner_surf_gifti=suit_assets["PIAL_FSL.surf.gii"],
        outer_surf_gifti=suit_assets["WHITE_FSL.surf.gii"],
    )
    return np.asarray(data).squeeze().astype(np.float32)


# =============================================================================
# Naming
# =============================================================================

def slugify(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "x"


def result_category_and_slug(source: Path, output_dir: Path) -> tuple[str, str]:
    """Split a result path into (category, informative_slug).

    category is the top-level result root (group_average / groupstats /
    noise_ceiling), used only to keep the metrics/ directory tidy. slug is
    every remaining path component (model/modality, config, result-file
    stem) joined with '__' -- this is what makes each exported filename
    self-describing without needing to open it or inspect its parent dirs.
    """
    try:
        rel = source.relative_to(output_dir)
    except ValueError:
        rel = Path(source.stem)
    parts = list(rel.parts)
    category = parts[0] if parts else "result"
    tail = parts[1:] if len(parts) > 1 else [source.stem]
    tail = list(tail)
    tail[-1] = tail[-1].removesuffix(".dscalar.nii")
    slug = "__".join(slugify(p) for p in tail)
    return category, slug


# =============================================================================
# Multi-map GIFTI I/O
# =============================================================================

def _save_multimetric(path: Path, maps: list[tuple[str, np.ndarray]]) -> None:
    arrays = []
    for name, values in maps:
        arr = GiftiDataArray(np.asarray(values, dtype=np.float32), intent="NIFTI_INTENT_SHAPE")
        arr.meta["Name"] = name
        arrays.append(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(GiftiImage(darrays=arrays), str(path))


def _assert_vertex_count(path: Path, expected: int) -> None:
    img = nib.load(str(path))
    n = img.darrays[0].data.shape[0]
    if n != expected:
        raise AssertionError(f"{path}: wrote {n} vertices, expected {expected} for its carrier mesh")


def _load_maps(path: Path, expected_columns: int) -> list[tuple[str, np.ndarray]]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != expected_columns:
        raise ValueError(f"{path}: {data.shape[1]} columns, expected {expected_columns}")
    names = [str(x) for x in img.header.get_axis(0).name]
    return [
        (names[i] if i < len(names) else f"map_{i}", data[i])
        for i in range(data.shape[0])
    ]


def discover_group_results(output_dir: Path) -> list[Path]:
    """Return every current group-level CIFTI, never per-subject intermediates."""
    roots = [output_dir / "group_average", output_dir / "groupstats", output_dir / "noise_ceiling"]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.dscalar.nii"))
    return sorted(set(paths))


def _add_to_spec(wb_command: Path, spec: Path, structure: str, filename: Path) -> None:
    subprocess.run(
        [str(wb_command), "-add-to-spec-file", str(spec), structure, str(filename)],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _make_boundary_assets(template: Path, bundle_dir: Path, wb_command: Path) -> dict:
    template_img = nib.load(str(template))
    bm_axis = template_img.header.get_axis(1)
    slices = struct_slices(bm_axis)
    surface_dir = bundle_dir / "surfaces" / "anatomical_units"
    component_cache = bundle_dir / "_component_cache"  # internal only, never browsed/loaded directly
    part_assets = {}

    needed_parts = {
        part
        for cfg in ANATOMICAL_GROUPS.values()
        if cfg["surface"] == "boundary"
        for part in cfg["parts"]
    }
    for part in sorted(needed_parts):
        info = slices[part]
        surf = component_cache / f"{part}.surf.gii"
        inc_path = component_cache / f"{part}.vertex_voxels.npz"
        if not surf.exists() or not inc_path.exists():
            coords, faces, incidence = boundary_mesh(info["voxel_ijk"], bm_axis.affine)
            save_surface(surf, coords, faces)
            inc_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(inc_path, vertex_voxels=incidence)
        part_assets[part] = {"surface": str(surf), "incidence": str(inc_path)}

    group_assets = {}
    for group, cfg in ANATOMICAL_GROUPS.items():
        if cfg["surface"] != "boundary":
            continue
        coords_all, faces_all, offset = [], [], 0
        for part in cfg["parts"]:
            coords, faces = load_surface(Path(part_assets[part]["surface"]))
            coords_all.append(coords)
            faces_all.append(faces + offset)
            offset += len(coords)
        surface = surface_dir / f"{group}.surf.gii"
        save_surface(surface, np.vstack(coords_all), np.vstack(faces_all))
        set_structure(wb_command, surface, cfg["carrier"], surface_type="ANATOMICAL")
        group_assets[group] = {
            "surface": str(surface),
            "parts": list(cfg["parts"]),
            "carrier": cfg["carrier"],
            "n_vertices": offset,
        }
    return {"parts": part_assets, "groups": group_assets, "slices": slices, "axis": bm_axis}


def _vertex_values(values: np.ndarray, incidence_file: str) -> np.ndarray:
    incidence = np.load(incidence_file, allow_pickle=True)["vertex_voxels"]
    out = np.full(len(incidence), np.nan, dtype=np.float32)
    for i, voxel_ids in enumerate(incidence):
        if len(voxel_ids):
            out[i] = np.nanmean(values[np.asarray(voxel_ids, dtype=np.int64)])
    return out


# =============================================================================
# Bundle build
# =============================================================================

def build_workbench_bundle(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    template: Path | None = None,
    bundle_dir: Path | None = None,
    wb_command: Path = DEFAULT_WB,
    result_files: list[Path] | None = None,
    skip_atlas_download: bool = True,
) -> Path:
    """Rebuild a complete, deterministic spec from all current group results."""
    output_dir = Path(output_dir)
    template = Path(template or output_dir / "subcortical_template.dscalar.nii")
    bundle_dir = Path(bundle_dir or output_dir / "workbench_visualization")
    wb_command = Path(wb_command)
    result_files = discover_group_results(output_dir) if result_files is None else sorted(result_files)
    if not template.exists():
        raise FileNotFoundError(template)
    if not result_files:
        raise FileNotFoundError(f"No group-level .dscalar.nii results under {output_dir}")

    bundle_dir.mkdir(parents=True, exist_ok=True)
    assets = _make_boundary_assets(template, bundle_dir, wb_command)
    slices, bm_axis = assets["slices"], assets["axis"]
    suit_assets = copy_suit_assets(bundle_dir, wb_command)
    atlases = fetch_cerebellar_parcellations(
        bundle_dir, wb_command, suit_assets, skip_download=skip_atlas_download
    )

    spec = bundle_dir / "subcortical_wb_view.spec"
    spec.unlink(missing_ok=True)
    for name in ("FLAT.surf.gii", "PIAL_FSL.surf.gii", "WHITE_FSL.surf.gii",
                 "PIAL_SUIT.surf.gii", "WHITE_SUIT.surf.gii"):
        _add_to_spec(wb_command, spec, "CEREBELLUM", Path(suit_assets[name]))
    if "SUIT.shape.gii" in suit_assets:
        _add_to_spec(wb_command, spec, "CEREBELLUM", Path(suit_assets["SUIT.shape.gii"]))
    for group in ANATOMICAL_GROUPS:
        if group != "CEREBELLUM":
            item = assets["groups"][group]
            _add_to_spec(wb_command, spec, item["carrier"], Path(item["surface"]))
    for atlas in atlases.values():
        _add_to_spec(wb_command, spec, "CEREBELLUM", Path(atlas["label"]))
        _add_to_spec(wb_command, spec, "CEREBELLUM", Path(atlas["border"]))

    exported, skipped = [], []
    for source in result_files:
        try:
            maps = _load_maps(source, len(bm_axis))
        except Exception as exc:
            log.warning("Skipping %s: %s", source, exc)
            skipped.append({"source": str(source), "reason": str(exc)})
            continue
        category, slug = result_category_and_slug(source, output_dir)
        result_dir = bundle_dir / "metrics" / category
        map_names = [m[0] for m in maps]

        cerebellar_maps = []
        for map_name, data in maps:
            volume = np.full(bm_axis.volume_shape, np.nan, dtype=np.float32)
            for part in ANATOMICAL_GROUPS["CEREBELLUM"]["parts"]:
                info = slices[part]
                ijk = info["voxel_ijk"]
                volume[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = data[info["start"]:info["stop"]]
            vol_img = nib.Nifti1Image(volume, bm_axis.affine)
            cerebellar_maps.append((map_name, map_cerebellum_to_suit(vol_img, suit_assets)))
        cerebellum_path = result_dir / f"{slug}__CEREBELLUM.func.gii"
        _save_multimetric(cerebellum_path, cerebellar_maps)
        set_structure(wb_command, cerebellum_path, "CEREBELLUM")
        _add_to_spec(wb_command, spec, "CEREBELLUM", cerebellum_path)

        result_exports = {"CEREBELLUM": str(cerebellum_path)}
        for group, cfg in ANATOMICAL_GROUPS.items():
            if group == "CEREBELLUM":
                continue
            group_maps = []
            for map_name, data in maps:
                parts = []
                for part in cfg["parts"]:
                    info = slices[part]
                    values = data[info["start"]:info["stop"]]
                    parts.append(_vertex_values(values, assets["parts"][part]["incidence"]))
                group_maps.append((map_name, np.concatenate(parts)))
            metric = result_dir / f"{slug}__{group}.func.gii"
            _save_multimetric(metric, group_maps)
            _assert_vertex_count(metric, assets["groups"][group]["n_vertices"])
            set_structure(wb_command, metric, cfg["carrier"])
            _add_to_spec(wb_command, spec, cfg["carrier"], metric)
            result_exports[group] = str(metric)
        exported.append({
            "source": str(source), "category": category, "slug": slug,
            "maps": map_names, "files": result_exports,
        })

    manifest = {
        "format_version": 3,
        "template": str(template),
        "spec": str(spec),
        "policy": "one cerebellum, one brain stem, and one bilateral file per paired structure; "
                  "filenames fully encode the analysis (category__model_modality__config__result__STRUCTURE.func.gii)",
        "groups": ANATOMICAL_GROUPS,
        "results": exported,
        "skipped": skipped,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (bundle_dir / "README_wb_view.md").write_text(
        "# Subcortical Workbench bundle\n\n"
        f"Open the movie scene, then import `{spec}`.\n\n"
        "The spec contains the SUIT cerebellar surfaces and one independent mesh for "
        "brain stem and every bilateral subcortical structure. Each result exports one "
        "`.func.gii` per structure, named "
        "`<category>__<model_modality>__<config>__<result>__<STRUCTURE>.func.gii` so the "
        "file alone identifies the analysis; it holds every diagnostic map from that "
        "result (e.g. searchlight_rho, sigmap_fdr) as named internal maps. To add a new "
        "analysis manually (e.g. a future omni3b run), drop its `.func.gii` next to the "
        "existing ones for that structure and load it in wb_view -- it is guaranteed "
        "vertex-compatible with the already-loaded mesh. Dense all-subcortical CIFTIs "
        "and whole-hemisphere meshes are deliberately excluded.\n"
    )
    log.info("Built %s with %d result sets (%d skipped)", spec, len(exported), len(skipped))
    return spec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--template", type=Path, default=None)
    p.add_argument("--bundle-dir", type=Path, default=None)
    p.add_argument("--wb-command", type=Path, default=DEFAULT_WB)
    p.add_argument("--download-atlases", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_workbench_bundle(
        output_dir=args.output_dir,
        template=args.template,
        bundle_dir=args.bundle_dir,
        wb_command=args.wb_command,
        skip_atlas_download=not args.download_atlases,
    ))


if __name__ == "__main__":
    main()
