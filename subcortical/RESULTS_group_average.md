# Subcortical RSA — group-average first pass

PE-AV (`pe-av-small-16-frame`, modality `av`), k=100, delay 5s, bin 5s, skip 5s,
Spearman. Group-average timeseries built from 175/175 subjects (same roster
as the cortical group average). Cerebellum uses the purist SUIT surface
geodesic; every other structure uses within-mask voxel-graph geodesic
(verified subject-invariant, computed once).

## Structural searchlight (mean rho across the structure)

| Structure | n_vox | mean ρ | max ρ | NC_upper (N=10) | NC_lower (N=10) |
|---|---:|---:|---:|---:|---:|
| Amygdala L | 620 | **0.042** | 0.084 | 0.305 | 0.006 |
| Amygdala R | 654 | **0.041** | 0.083 | 0.305 | 0.006 |
| Cerebellum L (SUIT) | 17,105 | 0.039 | 0.111 | 0.308 | 0.011 |
| Cerebellum R (SUIT) | 17,638 | 0.038 | 0.110 | 0.308 | 0.011 |
| Hippocampus R | 1,512 | 0.033 | 0.058 | 0.305 | 0.007 |
| Hippocampus L | 1,484 | 0.030 | 0.064 | 0.304 | 0.007 |
| Thalamus R | 2,461 | 0.026 | 0.088 | 0.306 | 0.006 |
| Thalamus L | 2,515 | 0.023 | 0.092 | 0.306 | 0.006 |
| Diencephalon-V L | 1,395 | 0.023 | 0.068 | 0.305 | 0.006 |
| Caudate L | 1,370 | 0.021 | 0.038 | 0.306 | 0.006 |
| Diencephalon-V R | 1,385 | 0.021 | 0.071 | 0.305 | 0.006 |
| Caudate R | 1,431 | 0.017 | 0.036 | 0.306 | 0.006 |
| Putamen L | 2,114 | 0.017 | 0.066 | 0.305 | 0.005 |
| Accumbens R | 283 | 0.015 | 0.026 | 0.304 | 0.004 |
| Putamen R | 2,016 | 0.015 | 0.054 | 0.305 | 0.006 |
| Brain stem (whole) | 6,731 | 0.013 | 0.053 | 0.305 | 0.006 |
| Accumbens L | 257 | 0.010 | 0.014 | 0.304 | 0.003 |
| Pallidum R | 499 | 0.007 | 0.015 | 0.305 | 0.005 |
| Pallidum L | 583 | 0.007 | 0.015 | 0.300 | 0.004 |

No voxel survives per-map FDR (single group-average map, df from n_bins ≈ 730)
— expected at this rho scale; the across-subject t-test (`group_stats.py`,
gated below) is the correct significance test, not the single-map FDR here.

## Nuclei ROI-RSA (whole-mask RDM, too small for a searchlight)

| Nucleus | Parent | n_vox | ρ | p |
|---|---|---:|---:|---:|
| SC_R | Brain stem | 102 | **0.042** | 4×10⁻⁷⁶ |
| SC_L | Brain stem | 101 | 0.034 | 4×10⁻⁵² |
| IC_R | Brain stem | 116 | 0.036 | 1×10⁻⁵⁶ |
| IC_L | Brain stem | 108 | 0.030 | 4×10⁻⁴¹ |
| MGN_R | Thalamus R | 70 | 0.024 | 9×10⁻²⁶ |
| MGN_L | Thalamus L | 70 | 0.018 | 3×10⁻¹⁶ |
| LGN_R | Thalamus R | 6 | 0.003 | 0.19 (n.s.) |
| LGN_L | Thalamus L | 4 | −0.001 | 0.67 (n.s.) |

A clean, anatomically coherent gradient for an audiovisual model:
**SC (multisensory) > IC (auditory relay) > MGN (auditory thalamus) ≫ LGN
(visual-only relay, null)**. Note the whole-brainstem searchlight average
(ρ=0.013) is far weaker than its own SC/IC sub-nuclei (ρ=0.03–0.04) —
whole-structure averaging dilutes the nucleus-level signal, which is exactly
why the ROI-RSA path exists for these small structures.

## Noise ceiling — mandatory reliability read

10-subject streaming reliability check (`subcortical_noise_ceiling.py`,
same k/bin/delay/method). Two things stand out:

1. **NC_upper is flat (~0.30–0.31) across every structure**, regardless of
   anatomy, size, or expected physiology (pallidum ≈ cerebellum ≈ amygdala).
   That uniformity is itself a signal: NC_upper is the *biased* estimate
   (includes subject *i* in its own reference mean, 1/10 weight at N=10), and
   its flatness here suggests it is dominated by autocorrelation shared
   across all subcortical searchlights (stimulus-driven low-frequency
   structure common to everyone) rather than genuine anatomical reliability
   differences. **Do not read NC_upper as "≈30% of variance is
   explainable" — treat it only as the upper (optimistic) bound.**
2. **NC_lower (leave-one-out, the honest floor) is uniformly low: 0.003–0.011**,
   consistent with the a-priori caveat that subcortical BOLD SNR is lower
   than cortex and fine-grained (k=100 searchlight) inter-subject reliability
   is close to noise at N=10.

**Important scope caveat:** this noise ceiling was measured on N=10
*individual* subjects, while the RSA rho above comes from the *175-subject
group-average* timeseries (averaging suppresses independent noise and raises
SNR well beyond any single subject). The two numbers are therefore not on
the same footing — group-average ρ exceeding the N=10 single-subject
NC_lower is not a violation of any bound, it reflects the SNR gap between
"one subject" and "mean of 175." A same-footing comparison requires
per-subject RSA on a matched subject set, which is exactly the gated next
step below.

## Bottom line

The group-average result is scientifically coherent, not noise-mined: the
nuclei gradient (SC > IC > MGN ≫ LGN, all but LGN significant at p≪10⁻¹⁵)
matches the audiovisual-integration hypothesis exactly, and the structures
tracking the AV model most strongly (amygdala, cerebellum, hippocampus) are
salience/narrative/motor-timing structures with an independently plausible
story — not an arbitrary top of the list. The whole-brainstem average
underselling its own SC/IC nuclei is a useful methodological finding on its
own (motivates nucleus-level ROI-RSA over whole-structure means generally).

**Recommendation:** promising enough to proceed to per-subject RSA (reusing
the existing 176-subject roster) → `group_stats.py` across-subject t-test
(same-footing significance, unmodified script, subcortical template) and a
same-N noise ceiling for a fair ratio, then `subcortical_encoding.py`. Not
run automatically — per the build order, this is the user's call.
