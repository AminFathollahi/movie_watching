"""
cf_modeling/lib/subject_adapter.py
====================================
Minimal mock subject object compatible with vicsompy's MssCf interface.

MssCf uses exactly two attributes from the subject object passed to its
constructor:

    subject.subject  — string identifier (stored in the output DataFrame and
                       used to label log messages)
    subject.out_csv  — output directory where MssCf.saveout() writes npy files

No data paths, CIFTI loading, or HCP directory structure is needed.  This
adapter lets CfModel be initialised without an HcpSubject (which would require
Hedger et al.'s specific directory layout and HCP data to be accessible).
"""

import os
from pathlib import Path


class CfSubjectAdapter:
    """Minimal subject object for vicsompy's MssCf.

    Parameters
    ----------
    subject_id : str
        Short identifier for this subject/analysis run (e.g. "group_average",
        "100610").  Stored as self.subject.
    out_csv    : str | Path
        Absolute path to the directory where MssCf.saveout() will write
        npy files (betas_*.npy, train_scores.npy, test_scores.npy,
        best_alphas.npy, params.csv).  Created on construction if absent.

    Attributes
    ----------
    subject : str     — identifier passed through to the output DataFrame.
    out_csv : str     — absolute path to the output directory (str, not Path,
                        because os.path.join in MssCf expects str).
    out_flat : str    — vicsompy also uses out_flat for PNG outputs; we point
                        it at the same directory (not used in no-splicing mode).
    analysis_name : str  — set to subject_id as a placeholder; overwritten by
                           MssCf.__init__ if a different analysis_name is passed.
    """

    def __init__(self, subject_id: str, out_csv: str | Path) -> None:
        self.subject       = str(subject_id)
        self.out_csv       = str(Path(out_csv).resolve())
        self.out_flat      = self.out_csv   # vicsompy reads this attribute in prepare_out_dirs
        self.analysis_name = str(subject_id)
        os.makedirs(self.out_csv, exist_ok=True)

    def __repr__(self) -> str:
        return f"CfSubjectAdapter(subject={self.subject!r}, out_csv={self.out_csv!r})"
