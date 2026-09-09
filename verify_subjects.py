import os
from pathlib import Path

# Paths based on your terminal outputs
subjects_file = Path("/home/amin/Research/Representation/Movie/data/subjects.txt")
cifti_dir = Path("/media/amin/ADATA HD710 PRO/Research/Representation/Movie/data/individual-59k")
mid_dir = Path("/home/amin/Research/Representation/Movie/data/midthickness_1.6")

# 1. Load subjects (ignoring empty lines and comments)
subjects = []
with open(subjects_file, 'r') as f:
    for line in f:
        line = line.strip()
        # Skip empty lines and lines starting with '#'
        if line and not line.startswith('#'):
            # Grab just the subject ID (handles potential inline comments)
            sub = line.split()[0]
            subjects.append(sub)

print(f"Loaded {len(subjects)} subjects from subjects.txt...\n")

# Tracking missing files
missing_cifti = {sub: [] for sub in subjects}
missing_mid = {sub: [] for sub in subjects}
complete_count = 0

# Mapping run to phase
cifti_phases = {1: "AP", 2: "PA", 3: "PA", 4: "AP"}

# 2. Check files for each subject
for sub in subjects:
    is_complete = True
    
    # Check midthickness (L and R)
    for hem in ["L", "R"]:
        fname = f"{sub}.{hem}.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii"
        if not (mid_dir / fname).exists():
            missing_mid[sub].append(fname)
            is_complete = False
            
    # Check 4 raw CIFTI files
    for run_id, phase in cifti_phases.items():
        fname = f"{sub}_tfMRI_MOVIE{run_id}_7T_{phase}_Atlas_1.6mm_hp2000_clean.dtseries.nii"
        if not (cifti_dir / fname).exists():
            missing_cifti[sub].append(fname)
            is_complete = False
            
    if is_complete:
        complete_count += 1

# 3. Print Report
print("=" * 60)
print("VERIFICATION REPORT")
print("=" * 60)
print(f"Total subjects checked:           {len(subjects)}")
print(f"Subjects with ALL files present:  {complete_count}")
print(f"Subjects missing files:           {len(subjects) - complete_count}")
print("=" * 60)

if complete_count < len(subjects):
    print("\n--- DETAILS OF MISSING FILES ---")
    for sub in subjects:
        if missing_mid[sub] or missing_cifti[sub]:
            print(f"\nSubject: {sub}")
            for m in missing_mid[sub]:
                print(f"  [MISSING MIDTHICKNESS] {m}")
            for c in missing_cifti[sub]:
                print(f"  [MISSING CIFTI]        {c}")
else:
    print("\nAll required CIFTI and Midthickness files are present.")