# CKA: searchlight centered kernel alignment

Compares the window-by-window geometry of cortical responses in the HCP 7T movie-watching data with that of model embeddings, using centered kernel alignment (CKA; Kornblith et al. 2019). CKA compares centered inner products rather than rank-ordered distances; the cross-validated variants used here remove the bias that measurement noise adds to the brain's Gram matrix. Every map is a searchlight over the 108,441 cortical grayordinates with the k = 100 nearest geodesic neighbors. The audio (A), video (V) and joint (J) embeddings of each model are compared with the brain, and semi-partial and commonality maps isolate what each reproduces beyond the others.

## Method

### Gram matrices

Windows and binning are those of `rsa/`: K windows of B seconds (K = 626 at 5 s), responses shifted by 5 s and z-scored per run, each embedding channel centered (`--feature-scaling center`) or z-scored (`zscore`) per run. With X the (K, d) embedding and H = I − 11ᵀ/K, the model Gram matrix is G = H X Xᵀ H, scaled to unit Frobenius norm. For centered (K, K) matrices, CKA(A, B) = ⟨A, B⟩_F / (‖A‖_F ‖B‖_F).

### Cross-validated CKA (`cv`)

Subjects are split into M = 25 disjoint groups (subject i to group i mod M). Each group's responses are binned with the per-run mean removed, averaged and divided by the per-run standard deviation of the all-subject average. For a searchlight of P grayordinates, with Y_m the (K, P) responses of group m,

G_cv = Σ_{m≠n} Y_m Y_nᵀ / (M (M − 1) P).

Group noise is independent, so G_cv estimates the stimulus-driven second-moment matrix without bias. The map value, CKA(H G_cv H, G), can be negative. This is the whitened unbiased RDM cosine of Diedrichsen et al. (2021, eqs. 20–21) for independent noise, with subject groups as partitions.

### Whitened cross-validated CKA (`cv-ar`)

Noise is autocorrelated within runs. Its window covariance is Σ = (1 / ((M − 1) V)) Σ_v Σ_m (y_vm − ȳ_v)(y_vm − ȳ_v)ᵀ, with y_vm the K responses of grayordinate v in group m, ȳ_v their mean over groups and V the number of grayordinates. W is the symmetric inverse square root of HΣH, with the constant direction removed and eigenvalues floored at 10⁻³ of their mean. The map value is CKA(W G_cv W, W Xc (W Xc)ᵀ), Xc being the centered embedding: the whitened unbiased RDM cosine with noise covariance Σ (Diedrichsen et al. 2021, section 2.13). `cv` is the case Σ = I.

### Non-cross-validated CKA (`noncv`)

CKA(H Y Yᵀ H, G) on the group-average responses keeps the noise bias and is written as a diagnostic only, with all windows and with the four repeated clips dropped.

### Semi-partial correlation

Let c = (c_A, c_V, c_J) be the CKA values of the brain matrix with the three model Gram matrices and R the 3 × 3 matrix of cosines among those matrices. The correlation of the brain matrix with the residual of G_x after least-squares regression on the other two component matrices is

sp_x = [R⁻¹ c]_x / √[R⁻¹]_xx.

Nothing is fitted to the brain; for `cv-ar`, R uses the whitened matrices. A component in the span of the other two (residual norm below 10⁻⁶) gets a zero map.

### Commonality

The squared multiple correlation of the brain matrix with the component matrices in a subset S of {A, V, J} is R²(S) = c_Sᵀ R_SS⁻¹ c_S. The seven R²(S) are split into `unique_a`, `unique_v`, `unique_j`, `shared_av_only`, `shared_aj_only`, `shared_vj_only` and `shared_avj` by the inclusion–exclusion of the encoding partition (commonality analysis; Seibold & McPhee 1979; on model RDMs, Groen et al. 2018 and Hebart et al. 2018). `shared_av` = R²(A) + R²(V) − R²(AV), and `unique_j` = sp_j². Regions can be negative. R²(S) is quadratic in the CKA values, so zero-mean noise in them adds a positive bias, larger for single subjects than for the group average; the semi-partial maps, linear in c, have no such bias.

### Subject maps and random effects

For subject s, with Y_s binned like a group and S₋s the summed responses of the other N − 1 subjects, T_s = sym(Y_s S₋sᵀ) / P is the two-partition case of G_cv (as in the noise-ceiling lower bound of Nili et al. 2014). The subject's `cka_a`, `cka_v`, `cka_j` are CKA(H T_s H, G) for `cv` and CKA(W T_s W, W G W) for `cv-ar`, with the group whitener W; semi-partial and commonality maps follow from them. Over subjects, the mean, the standard error and a one-sample t (N − 1 degrees of freedom) are written, on Fisher z = arctanh of the values for CKA and semi-partial maps and on raw values for commonality maps, which are not correlations. The CKA file also holds the mean and t of z_J − z_A and z_J − z_V within subjects. Subject maps share the all-subject sum, so the t values are optimistic.

### Interaction decomposition

`cka/multimodal_decomposition.py` forms R = J − Ĵ, with Ĵ the ridge prediction of J from [A | V] (penalty by 5-fold cross-validation, residual on the fitted windows), and writes ‖R‖²_F / ‖J‖²_F and the non-cross-validated CKA of the brain with J, [A | V] and R per Glasser parcel and per searchlight.

## Usage

```bash
bash cka/analysis.sh partitions --bins 5
bash cka/analysis.sh run --models pe-av-small-16-frame nemotron_layer18_mp --bins 5
```

`partitions` builds the (M, K, 108,441) group-response array once per bin (about 7 GB at 5 s, held in memory by `run`); `run` writes the group maps. `subjects` writes the per-subject maps (it needs the whitener from `run`), and `aggregate` the subject semi-partial and commonality maps and the statistics over subjects. Main options: `--models`, `--bins`, `--variants center|zscore`, `--controls own|SET...` (A and V from the model itself, tag `unimodal_own`, or from a control set listed in `encoding/README.md`, tagged with its name). Roots are set in the `CONFIG` block of `cka/analysis.sh`; `python cka/cka_searchlight.py {partitions,run,subjects,aggregate,commonality} --help` lists all arguments.

## Outputs

Under `outputs/cka/raw/`, `{stem}` = `cka_59k_raw_k{k}_delay{D}s_bin{B}s_skip{B}s_{measure}_{scaling}`, with `{measure}` = `cv` or `cv-ar`.

| File | Contents |
|---|---|
| `group_average/{model}/{stem}_{tag}_models.dscalar.nii` | `cka_a`, `cka_v`, `cka_j` |
| `group_average/{model}/{stem}_{tag}_semipartial.dscalar.nii` | `sp_a`, `sp_v`, `sp_j` |
| `group_average/{model}/{stem}_{tag}_commonality.dscalar.nii` | `r2_a` … `r2_avj`, the seven regions, `shared_av` |
| `group_average/{model}/{stem}_{tag}_provenance.json` | sources, settings, definitions, component cosines |
| `group_average/{model}/diagnostics/` | `noncv` maps of J, with and without (`_norepeats`) the repeated clips |
| `subjects/{subject}/{model}/` | the `models`, `semipartial` and `commonality` maps of one subject |
| `subject_mean/{model}/{stem}_{tag}_{kind}[_sem].dscalar.nii` | `mean_{map}`, `sem_{map}` over subjects |
| `subject_mean/{model}/{stem}_{tag}_{kind}_random_effects.dscalar.nii` | `t_{map}`; for `models` also `diff_j_minus_a`, `t_j_minus_a`, `diff_j_minus_v`, `t_j_minus_v` |
| `intermediate/` | whitener, noise covariance and statistics, subject Gram norms |

## Current results (5 October 2026)

5-s windows, centered embeddings, k = 100. Group values are the 25-group maps; subject values are means over 175 subjects. All values are averages over the 108,441 cortical grayordinates.

| PE-AV, own A and V | R²(AVJ) | unique J | shared AV | r with encoding map |
|---|---|---|---|---|
| `cv`, group | 0.0065 | 0.0007 | 0.0013 | unique J 0.30, shared AV 0.66 |
| `cv`, subject mean | 0.0024 | 0.0003 | 0.0004 | |
| `cv-ar`, group | 0.0004 | 0.00004 | 0.0001 | unique J 0.24, shared AV −0.04 |

The `cv` shared AV map correlates 0.66 with the encoding shared AV map (0.58 with Whisper and PE-Core as A and V). After whitening (`cv-ar`) shared AV is close to zero and no longer resembles the encoding map. For Nemotron with its own A and V, the part shared by all three is negative on average (−0.0016). Subject means are 0.2 to 0.7 times the group values; noise inflates R²(S), but a single subject carries less stimulus-driven signal than a group of seven.

In subjects, the one-sample t of every unique part exceeds 3 at every grayordinate. Each subject value is a square plus a positive noise term, so this alone is not evidence of an effect; the spatial pattern is what carries information.

Across cortex, the group sp_J map correlates 0.41 (own A and V) and 0.47 (Whisper and PE-Core) with an audiovisual conjunction map (AV above both A and V) from an independent localizer experiment, and 0.33 and 0.47 after whitening. cka_J alone correlates 0.41; with cka_J partialled out, sp_J still correlates 0.15 and 0.24 (0.11 and 0.27 whitened). Spatial autocorrelation inflates these correlations, so no p values are given.

## Tests

`tests/test_cka.py`.
