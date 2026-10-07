# CKA: searchlight centered kernel alignment

Compares the window-by-window geometry of cortical responses in the HCP 7T movie-watching data with that of model embeddings, using centered kernel alignment (CKA; Kornblith et al. 2019). CKA compares centered inner products rather than rank-ordered distances. Every map is a searchlight over the 108,441 cortical grayordinates with the k = 100 nearest geodesic neighbors, computed for the group-average responses and for each subject's own responses. The audio (A), video (V) and joint (J) embeddings of each model are compared with the brain, and semi-partial and commonality maps isolate what each reproduces beyond the others.

## Method

### Gram matrices and CKA

Windows and binning are those of `rsa/`: K windows of B seconds (K = 626 at 5 s), responses shifted by 5 s, binned and z-scored per run, each embedding channel centered (`--feature-scaling center`) or z-scored (`zscore`) per run. With X the (K, d) embedding and H = I − 11ᵀ/K, the model Gram matrix is G = H X Xᵀ H, scaled to unit Frobenius norm. For a searchlight of P grayordinates with (K, P) responses Y, the brain matrix is B = H Y Yᵀ H and the map value is

CKA(B, G) = ⟨B, G⟩_F / (‖B‖_F ‖G‖_F).

B is not cross-validated, so it contains the measurement noise of Y as well as the stimulus-driven part, and CKA includes the bias this noise adds. B and G are positive semi-definite, so CKA is non-negative and above zero even for unrelated responses; the maps are compared across cortex, models and subjects rather than against zero. The group map uses the average of all subjects. A subject map uses that subject's responses alone, binned and z-scored per run, with the same G. J is also compared without the four repeated clips, as a diagnostic.

### Semi-partial correlation

Let c = (c_A, c_V, c_J) be the CKA values of the brain matrix with the three model Gram matrices and R the 3 × 3 matrix of cosines among those matrices. The correlation of the brain matrix with the residual of G_x after least-squares regression on the other two component matrices is

sp_x = [R⁻¹ c]_x / √[R⁻¹]_xx.

Nothing is fitted to the brain. A component in the span of the other two (residual norm below 10⁻⁶) gets a zero map.

### Commonality

Every double-centered matrix has entries that sum to zero, so the cosine between two such matrices equals the Pearson correlation of their K² entries. CKA values are therefore correlations of matrix entries: c_x correlates the entries of B with those of G_x, and R_xy those of G_x with G_y. For a subset S of {A, V, J}, regressing the entries of B on the entries of the component matrices in S by least squares, one weight per matrix, gives the ordinary squared multiple correlation

R²(S) = c_Sᵀ R_SS⁻¹ c_S,

the textbook expression from correlations (Cohen, Cohen, West & Aiken 2003). The seven R²(S) are split into `unique_a`, `unique_v`, `unique_j`, `shared_av_only`, `shared_aj_only`, `shared_vj_only` and `shared_avj` by inclusion–exclusion, exactly as in the encoding variance partition (commonality analysis; Seibold & McPhee 1979; on representational dissimilarity matrices, Groen et al. 2018 and Hebart et al. 2018). For example `unique_j` = R²(AVJ) − R²(AV) = sp_j², and `shared_av` = R²(A) + R²(V) − R²(AV). Regions can be negative.

Encoding's R² is the held-out variance of each grayordinate's time course explained by a ridge readout of all embedding dimensions; CKA's R² is the in-sample variance of the window-by-window similarity entries explained by one weight per model matrix. Besides the noise bias in c, R²(S) is quadratic in c, so variability in c adds a further positive bias, larger for single subjects than for the group average; the semi-partial maps, linear in c, have no such added bias.

### Pooling of subject maps

For every map, the mean over the N subjects and the standard error (standard deviation / √N) are written; for the CKA maps also the fraction of subjects with cka_J > cka_A and with cka_J > cka_V. No test against zero is made, since CKA is above zero for unrelated data.

### Noise ceiling

How high the subject-mean CKA can go given inter-subject variability is bounded as in Nili et al. (2014), with CKA in place of a rank correlation. In each searchlight, b_s = H Y_s Y_sᵀ H / ‖H Y_s Y_sᵀ H‖_F is the unit-norm brain matrix of subject s and B̄ the mean of the b_s over the N subjects. The upper bound is the mean over subjects of cos(b_s, B̄), which includes subject s in the reference; the lower bound is the mean of cos(b_s, B̄₋s), with B̄₋s = (N B̄ − b_s) / (N − 1) the mean of the other subjects. With ⟨b_s, B̄⟩ computed once per subject, ‖B̄₋s‖² = (N² ‖B̄‖² − 2N ⟨b_s, B̄⟩ + 1) / (N − 1)². The subject-mean CKA of a model that captured everything the subjects share would lie between the two bounds. Both bounds include the positive floor of CKA and any structure the subjects' matrices share that is not stimulus-driven, such as the within-run autocorrelation of the responses. The binned subject responses are cached as float16; all products are in float32.

### Interaction decomposition

`cka/multimodal_decomposition.py` forms R = J − Ĵ, with Ĵ the ridge prediction of J from [A | V] (penalty by 5-fold cross-validation, residual on the fitted windows), and writes ‖R‖²_F / ‖J‖²_F and the non-cross-validated CKA of the brain with J, [A | V] and R per Glasser parcel and per searchlight.

## Usage

```bash
bash cka/analysis.sh run --models pe-av-small-16-frame nemotron_layer18_mp --bins 5
bash cka/analysis.sh subjects --models pe-av-small-16-frame --controls own text-asr
bash cka/analysis.sh aggregate --models pe-av-small-16-frame --controls own text-asr
bash cka/analysis.sh noise-ceiling
```

`run` writes the group maps, `subjects` the maps of every listed subject (resumable; it reads the raw subject CIFTIs), `aggregate` the pooled subject maps, and `noise-ceiling` the bounds (it caches the binned subject responses, about 24 GB for 175 subjects, under `intermediate/subject_responses/`). Main options: `--models`, `--bins`, `--variants center|zscore`, `--controls own|SET...` (A and V from the model itself, tag `unimodal_own`, or from a control set listed in `encoding/README.md`, tagged with its name), `--limit N` (first N subjects). Roots are set in the `CONFIG` block of `cka/analysis.sh`; `python cka/cka_searchlight.py {run,subjects,aggregate,noise-ceiling} --help` lists all arguments.

## Outputs

Under `outputs/cka/raw/`, `{stem}` = `cka_59k_raw_k{k}_delay{D}s_bin{B}s_skip{B}s_noncv_{scaling}`.

| File | Contents |
|---|---|
| `group_average/{model}/{stem}_{tag}_models.dscalar.nii` | `cka_a`, `cka_v`, `cka_j` |
| `group_average/{model}/{stem}_{tag}_semipartial.dscalar.nii` | `sp_a`, `sp_v`, `sp_j` |
| `group_average/{model}/{stem}_{tag}_commonality.dscalar.nii` | `r2_a` … `r2_avj`, the seven regions, `shared_av` |
| `group_average/{model}/{stem}_{tag}_provenance.json` | sources, settings, definitions, component cosines |
| `group_average/{model}/diagnostics/` | map of J without the repeated clips (`_norepeats`) |
| `subjects/{subject}/{model}/` | the same four files for one subject |
| `subject_mean/{model}/{stem}_{tag}_{kind}.dscalar.nii` | `mean_{map}` over subjects; for `models` also `frac_j_gt_a`, `frac_j_gt_v` |
| `subject_mean/{model}/{stem}_{tag}_{kind}_sem.dscalar.nii` | `sem_{map}` |
| `subject_mean/{model}/{stem}_{tag}_provenance.json` | subjects, definitions |
| `noise_ceiling/cka_59k_raw_k{k}_delay{D}s_bin{B}s_skip{B}s_noncv_noise_ceiling.dscalar.nii` | `nc_lower`, `nc_upper` (with a `_provenance.json`) |
| `intermediate/subject_responses/` | binned, z-scored subject responses (float16) used by `noise-ceiling` |

## Current results (6 October 2026)

5-s windows, centered embeddings, k = 100. Group values use the average of 175 subjects; subject values are means over the 175 single-subject maps. All values are averages over the 108,441 cortical grayordinates.

| PE-AV, own A and V | R²(AVJ) | unique J | shared AV | r with encoding map |
|---|---|---|---|---|
| group | 0.0071 | 0.0007 | 0.0015 | unique J 0.30, shared AV 0.62 |
| subject mean | 0.0030 | 0.0003 | 0.0007 | unique J 0.26, shared AV 0.53 |

The group shared AV map correlates 0.62 with the encoding shared AV map (0.57 with Whisper and PE-Core as A and V). For Nemotron with its own A and V, the part shared by all three is negative on average (−0.0015). Subject means are 0.4 to 0.5 times the group values for the R² parts and 0.5 to 0.75 times for the CKA values: a single subject's responses are noisier than the average of 175, and noise in the brain matrix lowers its alignment with every model matrix.

The subject-mean cka_J is 0.045 (cka_A 0.036, cka_V 0.038). The noise ceiling of the subject-mean CKA averages 0.187 (lower bound; 5th to 95th percentile 0.088 to 0.392) and 0.216 (upper bound; 0.131 to 0.404), so the cortex mean of the subject-mean cka_J is 0.24 times that of the lower bound (cka_A 0.19, cka_V 0.20). Averaged over grayordinates, the fraction of subjects with cka_J above cka_A is 0.93 and above cka_V 0.92 (0.94 and 0.85 with Whisper and PE-Core as A and V).

Across cortex, the group sp_J map correlates 0.39 (own A and V) and 0.44 (Whisper and PE-Core) with an audiovisual conjunction map (AV above both A and V) from an independent localizer experiment. cka_J alone correlates 0.40; with cka_J partialled out, sp_J still correlates 0.15 and 0.21. Spatial autocorrelation inflates these correlations, so no p values are given.

## Tests

`tests/test_cka.py`.
