# CF modeling status — 2026-09-01

**2026-09-03 update:** the per-subject CF run
(`02_fit_cf_model.py --mode per_subject`, ROI pair
`cca_a_peav_1pct_lboe100` × `cca_p_peav_1pct_lboe100`) was deliberately
halted by the user to free the GPU, not a crash. 26 subjects are complete on
disk at `/media/amin/ADATA HD710 PRO/cf_modeling/per_subject/cca_a_peav_1pct_lboe100_cca_p_peav_1pct_lboe100/subjects/`
(each holding `band_sizes.npy`, `best_alphas.npy`, `betas_*.npy`,
`product_map.npy`, `product_map_nc.npy`, `R2_*.npy`, `R2_full.npy`,
`pipeline.log`), 8.6 GB total. Subject 140117 was killed mid-fit: its
directory holds only the `.yml` config and `pipeline.log`, no `.npy`, so it
will be re-fit rather than skipped when the driver resumes. Log:
`logs/cf_persubject_cca_peav_1pct_20260903_111148.log`. Resume is safe — the
driver skips subject directories that already contain outputs.

**2026-09-02 update:** `channel_cca_analysis.py` was rebuilt around a single
continuous-axes analysis (per-channel modality axis vs CCA axis, zero-order
and partial variants; no classes, no permutation testing, no FDR, no
winner-take-all). Every entry below this line describes the now-deleted
4-class/BH-FDR/median-difference framework and its artifacts (including the
+.077 headline number) for historical record only — see
`cf_modeling/channel_cca_analysis.py`'s module docstring and
`outputs/cf_modeling/channel_cca_preference/results/` for the current
results.

- **Completed:** ordinary CCA ROI-mean Pearson maps (`_raw_corr`) for all 11 prepared CCA pairs. Each pair has one 8-map dscalar plus bilateral, within-hemisphere, L, and R 32×32 bivariate dlabels (44 total). No bivariate `_nc` maps were made: `_nc` is split-CF R² minus the one-ROI-mean OLS null R², not ordinary correlation.
- **Completed:** PE-AV, Nemotron-18 MP, and Topo-Omni-18 sheet MP channel sensitivity from paired one-sided true-AV-minus-without-audio/video **block-bootstrap** tests (626 movie bins are autocorrelated, so the previous paired t-test's iid assumption was invalid; replaced with a moving-block bootstrap over `run_bins`, mirroring the block bootstrap already used for `compare_channel_classes`). Strict audio/video winner counts (unaffected by the test change — sign of the mean difference only) are PE-AV 477/547, Nemotron 1,010/1,038, and Topo-Omni 967/1,081. BH-FDR audio-only/video-only/both/neither counts under the block-bootstrap test are PE-AV 244/268/193/319, Nemotron 425/520/493/610, and Topo-Omni 366/503/553/626.
- **Analysis unit:** one full AV representation per model over 626 aligned 5-s bins: PE-AV 626×1,024 and both Omni families 626×2,048. Token-pool representations are out of scope and removed.
- **Multiplicity policy:** BH-FDR is retained because thousands of channel tests make raw p<.05 classifications invalid. BY and max-T FWER gates are not used to define exploratory channel labels; raw p-values, BH q-values, effect sizes, and confidence intervals remain available. Four-class cortical RSA separately reports empirical raw p, BH-FDR q, and max-T FWER p maps.
- **Full-embedding CCA result:** the prespecified PE-AV audio-only minus video-only partial CCA-A−CCA-P contrast (audio-only/video-only groups now drawn from the block-bootstrap channel classes above) is +.077, 95% block-bootstrap CI [.029, .119], BH q=.0024. Nemotron (+.011, q=.629) and Topo-Omni (−.008, q=.924) are not significant. Circular-shift permutation remains the correct null for the CCA correlation/connectivity tests themselves (a cross-correlation statistic) and was left unchanged.
- **Channel analysis policy:** each model now uses only its own top-1% CCA mask. PE-AV-global Nemotron/Topo-Omni variants and cross-mask comparison artifacts are deprecated and removed. Own-mask results are descriptive rather than independent localization.
- **Mask-dependence audit:** PE-AV vs own-model Dice is CCA-A/CCA-P=.855/.741 for Topo-Omni and .917/.800 for Nemotron. Because cross-mask results changed some channel contrasts, they are not pooled with the final analysis; each model now uses only its own non-independent top-1% descriptive mask.
- **Channel audit rebuild:** three own-mask result sets are consolidated in one executable notebook and one `results/` tree: PE-AV full, Nemotron-18 MP full, and Topo-Omni-18 sheet MP full. The notebook is `cf_modeling/channel_cca_preference.ipynb`.
- **Category RSA scope:** channel-class RSA uses only the four BH significance classes in those three full representations; winner-take-all RSA maps are excluded. Each class retains rho, empirical raw p, BH q, max-T FWER p, signed significance maps, and explicit raw/BH/max-T masks at .05. The synchronized null uses nonzero circular shifts within movie runs.
- **Prepared, not fit:** top-1% Nemotron-18 MP and Topo-Omni-18 MP/LT; AMPLE-70/75/80/90 PE-AV, Nemotron-18 MP, and Topo-Omni-18 LT; AMPLE-75/80/90 Topo-Omni-18 MP. Topo-Omni MP AMPLE-70 remains excluded because its right-hemisphere candidates overlap by 221 vertices.
- **Later fits:** default 100 requested LBOEs for the AMPLE grid; Nemotron-18 MP top-1% at 50/100 via `cca_nemotron_18_mp_lboe_sensitivity`. No new CF fits were started.
- **Earlier completed:** PE-AV top-1% 50/100/200 LBOE sensitivity fits and the per-ROI `_lboe200` artifact migration. Manifest: `/home/amin/Research/Representation/Movie/outputs/cf_modeling/lboe_naming_migration_manifest.json`.
