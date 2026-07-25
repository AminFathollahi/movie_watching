"""rsa/localizer_naming.py
Shared name construction for the Ward's-linkage sheet-localizer family
(topoomni_sheet_localizer.py's auditory/integration branches and
topoomni_av_separability_localizer.py's condition-contrast branch), so every
folder/file name says which model actually drove clustering (driver) and
which model was actually scored/read-out (sheet) -- the driver and sheet
model are often different, so a single family-name prefix would be
ambiguous or wrong.

Convention: localizer_{kind}[_{design}]_drv-{driver}_sheet-{sheet}[suffix][_c{N}|_all]
  kind   : "speech" (auditory positive control, topoomni_sheet_localizer.py) |
           "av_separability" (condition-contrast Fisher's-exact enrichment
           test, topoomni_av_separability_localizer.py)
  design : ONLY meaningful for kind="av_separability" -- which artificial
           condition set was clustered: "dummy" (default, no tag for
           backward compatibility with pre-existing outputs -- 3-condition
           av/clsav_from_a/clsav_from_v, real pairing vs unimodal+blank) or
           "scramble" (2-condition av/avscramble, real vs temporally-
           mismatched pairing -- tests binding/synchrony, not modality
           presence). Pass design="scramble" to tag scramble-design outputs;
           leave design="" (default) for the original dummy-modality design.
  driver : short tag for the independent embedding that drove Ward's-linkage
           clustering (e.g. "text", "transcript", "whisper", "topoomni", "peav")
  sheet  : short tag for the model whose cortical-sheet units were actually
           scored/read-out (e.g. "topoomni", "peav")
  suffix : tags a unit-selection-mode variant (e.g. "_fdr"); default "" keeps
           the top-1% default's names byte-identical to pre-existing runs.
"""

KINDS = ("speech", "av_separability")
DESIGNS = ("", "dummy", "scramble")  # "" and "dummy" are equivalent (no tag)

# Short tags for models that can appear as either driver or scored sheet.
# TopoOmni's mean-pool ("_mp") and last-token ("_lt") sheet readouts are
# genuinely different embeddings (see nemotron_extract_intact.py's docstring
# on why both exist) and MUST get distinct tags -- collapsing them to one
# "topoomni" tag would silently overwrite one variant's output with the other's.
MODEL_TAG = {
    "topoomni_layer18_sheet_mp": "topoomni_mp",
    "topoomni_layer18_sheet_lt": "topoomni_lt",
    "pe-av-small-16-frame": "peav",
    # Keep the pooling suffix ("_mp"/"_lt") in the tag, matching topoomni's
    # shape below -- a bare "nemotron" tag would collide between the
    # mean-pool and last-token readouts, silently overwriting one variant's
    # output with the other's.
    "nemotron_layer36_mp": "nemotron_mp",
    "nemotron_layer36_lt": "nemotron_lt",
}

# Short tags for (model, modality) pairs used as the independent clustering
# driver in topoomni_sheet_localizer.py (--cluster-embedding-model/-modality).
DRIVER_MODALITY_TAG = {
    ("pe-av-small-16-frame", "event_t"): "text",
    ("pe-av-small-16-frame", "transcript_t"): "transcript",
    ("whisper-large-v3", "a"): "whisper",
}


def model_tag(model_name: str) -> str:
    return MODEL_TAG.get(model_name, model_name)


def driver_tag(model_name: str, modality: str | None = None) -> str:
    if modality is not None:
        return DRIVER_MODALITY_TAG.get((model_name, modality), f"{model_name}_{modality}")
    return model_tag(model_name)


def _design_tag(design: str) -> str:
    assert design in DESIGNS, f"design must be one of {DESIGNS}, got {design!r}"
    return f"_{design}" if design and design != "dummy" else ""


def base_name(kind: str, driver: str, sheet: str, suffix: str = "", design: str = "") -> str:
    """`suffix` tags a variant of the same driver/sheet run so it does not
    clobber the default. Used for the "_fdr" unit-selection variant, which
    must coexist with the default top-k readout. suffix="" (default) keeps the
    original names byte-identical. `design` (av_separability only) tags which
    condition set was clustered -- see module docstring; "" or "dummy" both
    mean the original 3-condition design and add no tag."""
    assert kind in KINDS, f"kind must be one of {KINDS}, got {kind!r}"
    return f"localizer_{kind}{_design_tag(design)}_drv-{driver}_sheet-{sheet}{suffix}"


def summary_json_name(kind: str, driver: str, sheet: str, suffix: str = "", design: str = "") -> str:
    """"speech" has one clustering run and one JSON per driver/sheet pair;
    "av_separability" has its own JSON. `suffix`/`design` as in base_name()."""
    if kind == "av_separability":
        return f"_localizer_av_separability_summary{_design_tag(design)}_drv-{driver}_sheet-{sheet}{suffix}.json"
    return f"_localizer_summary_drv-{driver}_sheet-{sheet}{suffix}.json"


def demo():
    assert model_tag("topoomni_layer18_sheet_lt") == "topoomni_lt"
    assert model_tag("topoomni_layer18_sheet_mp") == "topoomni_mp"
    assert model_tag("pe-av-small-16-frame") == "peav"
    assert model_tag("nemotron_layer36_mp") == "nemotron_mp"
    assert model_tag("nemotron_layer36_lt") == "nemotron_lt"
    assert driver_tag("whisper-large-v3", "a") == "whisper"
    assert driver_tag("pe-av-small-16-frame", "event_t") == "text"
    assert driver_tag("topoomni_layer18_sheet_lt") == "topoomni_lt"
    assert base_name("speech", "whisper", "topoomni_lt") == "localizer_speech_drv-whisper_sheet-topoomni_lt"
    assert summary_json_name("speech", "whisper", "peav") == "_localizer_summary_drv-whisper_sheet-peav.json"
    assert base_name("av_separability", "peav", "topoomni_mp") == "localizer_av_separability_drv-peav_sheet-topoomni_mp"
    assert base_name("av_separability", "peav", "topoomni_lt", design="dummy") == "localizer_av_separability_drv-peav_sheet-topoomni_lt"
    assert base_name("av_separability", "topoomni_lt", "peav", suffix="_fdr", design="scramble") == \
        "localizer_av_separability_scramble_drv-topoomni_lt_sheet-peav_fdr"
    assert summary_json_name("av_separability", "topoomni_lt", "peav", design="scramble") == \
        "_localizer_av_separability_summary_scramble_drv-topoomni_lt_sheet-peav.json"
    assert summary_json_name("av_separability", "topoomni_lt", "peav") == "_localizer_av_separability_summary_drv-topoomni_lt_sheet-peav.json"
    print("localizer_naming.py: all checks passed")


if __name__ == "__main__":
    demo()
