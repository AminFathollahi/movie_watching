import torch
from safetensors.torch import save_file
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import MODELS  # noqa: E402

# --- IMPORTANT: UPDATE THIS HASH ---
# Run 'ls <models folder>/manual/models--facebook--PE-Core-G14-448/snapshots/' 
# to find the correct hash folder name for G14.

conversions = [
    {
        "dir": str(MODELS / "manual/models--facebook--PE-Core-G14-448"),
        "old_name": "PE-Core-G14-448.pt"
    },
    {
        "dir": str(MODELS / "manual/models--laion--larger_clap_music_and_speech/snapshots/195c3a3e68faebb3e2088b9a79e79b43ddbda76b"),
        "old_name": "pytorch_model.bin"
    },
    {
        "dir": str(MODELS / "manual/models--shikhar7ssu--OpenBEATs-Large-i2/snapshots/186e34c71c01b39e29572bc047cfc100dda6289a/work/nvme/bbjs/sbharadwaj/7Msounds/exp/beats_iter1_large1.tune_lr1.0e-4_warmup40000_bins1600000_totalsteps400000"),
        "old_name": "epoch_latest.pt"
    }
]

for item in conversions:
    source_path = Path(item["dir"]) / item["old_name"]
    
    # Target always goes to the snapshot root for AutoModel compatibility
    if "snapshots" in item["dir"]:
        # Logic to find the snapshot root (the folder with the hash name)
        parts = Path(item["dir"]).parts
        snap_idx = parts.index("snapshots")
        target_dir = Path(*parts[:snap_idx + 2])
    else:
        target_dir = Path(item["dir"])

    target_file = target_dir / "model.safetensors"

    if source_path.exists():
        print(f"--- Converting {item['old_name']} ---")
        
        # Ensure the target directory exists before saving
        target_dir.mkdir(parents=True, exist_ok=True)
        
        state_dict = torch.load(source_path, map_location="cpu", weights_only=False)
        
        # Clean up nested state_dicts
        if isinstance(state_dict, dict):
            if "model" in state_dict: state_dict = state_dict["model"]
            elif "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        
        try:
            save_file(state_dict, str(target_file))
            print(f"Success: {target_file}")
        except Exception as e:
            print(f"Save failed: {e}")
    else:
        print(f"Skipped: {source_path} not found.")