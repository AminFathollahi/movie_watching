import os
from pathlib import Path

ROOT = Path(os.environ.get("MOVIE_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
EXTERNAL = ROOT / "external"
MODELS = Path(os.environ.get("MOVIE_MODELS_DIR", ROOT / "hf_models"))
