# Revised research and implementation plan

This document preserves the two Astra reviews that replaced the former Track B plan. Part I identifies the main methodological and interpretive gaps. Part II turns the additivity, compression, and clustering recommendations into an implementation plan.

---

## Part I — Review of the existing project and missing analyses

You already have substantial coverage—multiple model families, unimodal controls, residualization, scrambling, noise ceilings, and subject×stimulus inference. The biggest opportunity now is to sharpen what distinguishes **audiovisual integration from strong auditory alignment**, and make compression an explicit experimental variable.

After reading Track B, the results reports, and relevant code, I would prioritize the following.

### 1. Resolve the encoding comparison before treating the RSA–encoding disagreement as a neuroscientific finding

I found two concrete issues:

- `encoding/variance_partition.py` constructs AV residuals **before** splitting training and test stimuli. The residual function in `rsa/multimodal_decomposition.py` standardizes all samples, selects a ridge penalty, then fits and predicts on all samples. Cross-validation selects the penalty; it does not make those residuals out-of-fold. This uses held-out stimulus information, although it does not use held-out brain responses.
- `r2_score_split` allocates the joint model’s R² among feature spaces. It is not automatically the unique improvement from adding AV features, particularly when ridge residualization leaves correlations between bands. That distinction is explicit in the [Himalaya documentation](https://gallantlab.org/himalaya/_generated/himalaya.scoring.r2_score_split.html).

My first analysis would be a properly nested, held-out comparison:

\[
\Delta R^2 = R^2_{\mathrm{test}}(A+V+AV)-R^2_{\mathrm{test}}(A+V).
\]

Fit normalization, residualization, dimensionality reduction, and hyperparameters inside training folds. Evaluate paired differences on identical held-out clips. Start with anatomical auditory, posterior temporal, and frontal ROIs. This could materially change how you interpret the current discrepancy.

### 2. The scramble experiment needs an additive-model control to isolate binding

Your extraction code keeps video fixed and randomly replaces audio. Consequently, even a model with **no audiovisual interaction whatsoever** can lose brain alignment: it is now representing the wrong soundtrack.

The 500-seed result therefore establishes sensitivity to correct stimulus pairing, but does not by itself establish nonlinear integration.

A useful extension is to apply the *same pairings* to both the fused model and an additive A+V baseline, then compare their intact-to-scrambled losses. Evaluate this after accounting for aligned unimodal information. A larger fusion-specific loss would be more informative than the intact–scrambled difference alone.

For a smaller follow-up, distinguish temporal disruption within a clip from replacement across unrelated clips. The current manipulation changes synchrony, content, and narrative compatibility simultaneously.

### 3. Track B’s proposed “integration axis” is too permissive

`1 − R²(mean_av ~ mean_a + mean_v)` measures unexplained response. It can be high because of nonlinear interaction, but also because of unstable cluster means, omitted unimodal dimensions, or unusual responses to blank/silent inputs.

In particular, predicting a cluster’s AV mean using only that cluster’s two unimodal means is a weak additive baseline. Other unimodal channels may explain it perfectly.

I would retain the proposed measure as descriptive, call it **linear unexplained fraction**, and add prediction from the full regularized unimodal representations. Require that the unexplained component itself predicts held-out brain responses before interpreting it as brain-relevant integration.

There is also a logical mismatch in B1: the existing near-zero correlations test whether *modality preference tracks anterior–posterior seed preference*. The new spread gate tests whether *clusters differ in modality preference*. Those are different hypotheses. Passing the gate would not resolve the original null.

### 4. Do not let a cluster-mean null conclude that the distinction exists only between processing stages

The statement that modality preference is absent “at any granularity” would exceed what B1 tests. Distributed information can cancel in means while remaining readily available in a multivariate subspace.

Channel clustering is also sensitive to the representation’s coordinate system. A rotation can alter channels and their clusters while preserving Euclidean representational geometry.

This suggests a particularly useful comparison: **cluster means versus equally dimensional PCA or other training-fitted subspaces**. If a subspace predicts modality-related brain differences but cluster means do not, that is positive evidence about the representational format.

For B1’s uncertainty, bootstrap movie blocks as well as assessing membership sensitivity. Resampling channels within fixed clusters does not establish generalization to other stimuli. And in the B2 demo, a simple permutation of cluster *names* must give ARI=1; the null requires randomized memberships.

### 5. Make compression central to the scientific question

I did not find an explicit compression–alignment experiment in Track B. Whether “compressed” means reduced-dimensional embeddings or compact/distilled transformers, define the compression variable and compare it against a reference.

For dimensional compression, I would measure held-out alignment and incremental AV benefit across a modest dimension sweep, using:

- PCA learned from training stimuli;
- random projections at matched dimension;
- your cluster-mean representation at matched dimension;
- the full representation.

The interesting question becomes: **Does compression preserve audiovisual information beyond A+V, and does it preserve that information differently in anterior auditory versus posterior temporal cortex?**

This could be a stronger paper contribution than another cluster correspondence map. It also separates denoising benefits from evidence that the particular compression method captures meaningful organization.

### 6. Cross-model replication helps, but does not fully remove seed-selection bias

Nemotron was not used to define the PE-AV seeds, which is valuable. However, correlated models evaluated against the same stimulus and brain data can inherit selection advantages. Architecture independence is not data independence.

Add an anatomical ROI analysis, or define seeds on training subjects/stimuli and evaluate them on held-out data. Keep related participants together if using subject splits.

More importantly, directly test the proposed dissociation. “Anterior binds speech with music; posterior binds speech with faces” requires an interaction between **region and content relationship**, beyond speech, music, and face main effects. Category-map overlap alone cannot demonstrate binding.

### 7. B4 needs stronger separation of shared stimulus responses from coupling

Sliding-window correlations can rise when two regions respond to the same event, even without a change in their relationship. Regressing window means and global signal does not necessarily remove this. Task-evoked responses can systematically inflate functional-connectivity estimates. [Cole et al., 2019](https://www.colelab.org/pubs/ColeEtAl2019NeuroImage.pdf)

I would prefer an explicit interaction model containing seed activity, continuous annotations, their interaction terms, and appropriate stimulus-response/nuisance terms. Use run-aware temporal inference and examine HRF sensitivity. Treat group-average analyses as stimulus-locked cofluctuation evidence; stronger coupling claims need individual-level support. That need not require restarting the halted whole-cortex CF fits.

Also expand the annotation contract beyond 626 ordered rows: retain **actual run-local onset, offset, duration, and missingness**. Five-second labels cannot recover one-second content timing merely by interpolation, especially across omitted intervals.

### 8. A few HCP-specific checks could prevent expensive ambiguity later

The official protocol includes a common repeated validation clip, and HCP documents multiple stimulus versions. Verify each participant’s version/timing and keep repeated content appropriately grouped in validation. The repeated clip is also useful for checking timing and response reliability, although too short to carry the main inference. [HCP protocol](https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging), [stimulus-version documentation](https://wiki.humanconnectome.org/docs/7T%20Movie%20watching%20task%20clip%20info%20timing%20versions%20shown%20per%20individual%20subject.html)

There may also be useful existing annotations: published HCP work used `WordNetFeatures.hdf5` from the movie resources alongside brightness, optical flow, and face features. Those could support semantic and low-level controls while your richer labels are pending. [Finn and Bandettini, 2021](https://thefinnlab.github.io/files/Finn_NeuroImage2021.pdf)

One reporting issue deserves immediate attention. The September Markdown report treats 0.70% significance as a circular-shift result and describes several procedures as converging on the same question. The LaTeX report instead identifies 0.70% as an analytic time-bin test and reports approximately 99.8% for the actual circular-shift result. These are materially different interpretations. Also, the 4/8/16-block sensitivity should remain visible; finer blocking cannot be justified simply because it produces significance.

My preferred order would be: **repair and evaluate the held-out additive-versus-AV comparison; run matched compression curves in independently defined ROIs; then pursue content-conditioned anterior–posterior differences.** Keep Track B clustering as an exploratory characterization whose outcome can guide those tests.

---

## Part II — Concrete implementation design

Connect these into one experiment: **does a compact representation preserve brain-predictive audiovisual information that an additive model cannot explain?** Clustering then becomes one candidate compression method, with its interpretability tested separately.

There are three distinct claims to keep separate:

| Claim | Required evidence |
|---|---|
| AV features improve brain prediction beyond A+V | Held-out improvement over a strong additive baseline |
| That improvement depends on audiovisual interaction | A control that removes additive contributions while manipulating pairing |
| Clustering captures that information efficiently | Cluster representations preserve the improvement at low dimension, relative to matched alternatives |

### 1. Establish “AV beyond A+V” first

For each stimulus bin, define:

- \(A_t\): the model’s unimodal audio representation.
- \(V_t\): its unimodal video representation.
- \(J_t\): its intact joint AV representation.
- \(Y_t\): brain responses.

“A+V” means independently weighted feature spaces—not literally adding embedding vectors.

Fit two banded-ridge encoding models:

\[
M_0:\quad Y=AW_A+VW_V+\epsilon
\]

\[
M_1:\quad Y=AW_A+VW_V+JW_J+\epsilon.
\]

The outcome is:

\[
G=R^2_{\mathrm{test}}(M_1)-R^2_{\mathrm{test}}(M_0).
\]

For example, \(0.12-0.10=0.02\) means that adding joint features explains another two percentage points of held-out response variance.

Practically:

1. Hold out one movie run; train on the other three.
2. Select regularization using run-wise splits within those three.
3. Fit all learned feature transformations inside those splits.
4. Rotate through all four outer folds.
5. Save predictions and squared errors for every held-out bin, allowing paired comparisons by clip.

Keep the common repeated clip out of the main generalization analysis and use it separately for reliability. Report uncertainty over clips; for population-level claims, also evaluate individual participants and account for relatedness.

**No embedding residualization is necessary for this first comparison.** Directly adding \(J\) avoids the current residual-before-split problem. Use ordinary held-out R² differences, retaining negative values, rather than treating `r2_score_split` as an incremental effect. The latter allocates joint-model performance among bands. See the [Himalaya `r2_score_split` documentation](https://gallantlab.org/himalaya/_generated/himalaya.scoring.r2_score_split.html).

Start with PE-AV and Nemotron L18, anatomical auditory/posterior temporal ROIs, and the existing seeds as exploratory targets. Brain targets can include individual vertices within ROIs; averaging everything into one timeseries could discard the pattern information underlying the RSA result.

A positive \(G\) supports **additional linearly accessible information relative to these A/V features**. It does not yet establish a biological interaction. A stronger later baseline would allow nonlinear functions *within* each modality while remaining additive across modalities, for example an additive kernel model \(K_A+K_V\).

### 2. Test whether the benefit is specifically related to pairing

Comparing intact-to-scrambled losses for fused and additive models is useful, but differences in model quality and redundancy can complicate that comparison.

An easier-to-interpret first test keeps the correct unimodal information available throughout:

| Encoding model | Inputs |
|---|---|
| Baseline | Correct \(A_t,V_t\) |
| Intact extension | Correct \(A_t,V_t\), plus \(J_t=F(a_t,v_t)\) |
| Mismatched extension | Correct \(A_t,V_t\), plus \(J_t^{(s)}=F(a_{\pi_s(t)},v_t)\) |

For each mismatch seed:

\[
G_s=R^2_{\mathrm{test}}(A,V,J^{(s)})-R^2_{\mathrm{test}}(A,V).
\]

Then examine:

\[
B=G_{\mathrm{intact}}-\operatorname{mean}_s G_s.
\]

The question is: **does correctly paired fusion provide more additional predictive information than mismatched fusion, when the actual audio and video are already supplied?**

First use a predetermined subset of roughly 20–30 existing scramble seeds, then expand if the comparison is informative. The seed distribution measures sensitivity to mismatching; it is not automatically a valid permutation distribution for every integration claim.

Training examples must not receive held-out audio through the scramble. Reconstruct the pairing indices and use pairings confined to the relevant training or test partition. Existing global permutations may supply some usable examples, but cannot simply be used unchanged.

A stronger, more direct model-interaction test is also possible. Take two audiovisual segments \(i,j\), and form:

\[
D_{ij}=F(a_i,v_i)+F(a_j,v_j)-F(a_i,v_j)-F(a_j,v_i).
\]

If the representation is additive,

\[
F(a,v)=g(a)+h(v),
\]

then **\(D_{ij}=0\) exactly**, regardless of how complicated \(g\) and \(h\) are. Audio and video main effects cancel algebraically.

For practical brain alignment:

- Choose a fixed reference set of segments from training data.
- Compute \(D_{ij}\) for each target segment \(i\) against those references.
- Average over references to obtain an interaction representation \(I_i\).
- Test \(R^2_{\mathrm{test}}(A,V,I)-R^2_{\mathrm{test}}(A,V)\).

For training targets, use references from different training clips and exclude self/repeated-content pairs. For test targets, retain training references.

The existing scramble extractions may contain both crossed pairings needed for many quartets; that requires an availability/provenance audit. Otherwise, a targeted extraction could generate a balanced reference set.

Compute the contrast in a consistent representation space: input-dependent normalization can itself create nonadditivity, so ideally inspect a representation before final L2 normalization as well. Nonzero \(D\) establishes interaction **in the model**; its incremental brain prediction establishes relevance to brain responses. The HCP data do not contain experimentally mismatched brain responses, so this remains evidence of representational correspondence.

### 3. Turn compression into two complementary comparisons

Here compression means compression of extracted representations. Compact model size or distillation would be a separate experimental axis.

Use a small dimension grid such as:

\[
d\in\{2,4,8,16,32,64\},
\]

plus the full representation. For each outer training fold, construct:

| Method | Compressed representation |
|---|---|
| PCA | Projection onto \(d\) training-fitted components |
| Random projection | \(d\) random directions, averaged over predetermined seeds |
| Cluster means | \(d\) channel groups, represented by their means |
| Within-cluster PC1 | One training-fitted component per group |

Within-cluster PC1 is useful because channels with opposite signs can cancel in a mean while sharing a reliable signal.

#### A. Which representation predicts the brain most efficiently at a fixed feature budget?

Compare:

\[
R^2_{\mathrm{test}}(C_d(J))
\]

against an additive representation with the same total dimension, for example:

\[
R^2_{\mathrm{test}}(C_{d/2}(A),C_{d/2}(V)).
\]

Use a fixed equal allocation initially; if audio/video allocation is tuned, tune it inside training folds. Give both models equivalent regularization opportunities.

This is the **compression-efficiency** comparison.

#### B. How many dimensions retain information beyond a strong additive baseline?

Keep the same regularized A+V baseline across all dimensions, then measure:

\[
G_d=R^2_{\mathrm{test}}(A,V,C_d(J))-R^2_{\mathrm{test}}(A,V).
\]

Repeat with \(C_d(I)\) if the factorial interaction representation is available.

This is the **incremental-AV** comparison. Keeping A+V fixed is important: otherwise, the joint representation might appear useful simply because its unimodal competitors were compressed too aggressively.

The main artifact is a curve of \(G_d\) against dimension for each compression method and ROI, with uncertainty. Interpretations become concrete:

- Cluster means match full-representation \(G\) at small \(d\): compact groups preserve the useful information.
- PCA succeeds but cluster means fail: useful information is distributed in weighted combinations.
- Full \(J\) helps, but compressed \(J\) does not: the incremental information is fragile under those compression methods.
- Neither full nor compressed \(J\) helps: the encoding analysis does not establish additional value beyond the chosen baseline.

A “dimension needed to preserve 90%” summary is reasonable only if the full-representation improvement is reliably positive. Otherwise, that ratio is unstable.

### 4. Practical revision of Track B

Retain clustering, but change its selection, gates, and downstream tests.

#### B1: Learn reproducible channel groups and test their usefulness

Start with one primary clustering method: correlation-based/spherical k-means on channel timeseries, with a modest \(k\) grid such as 2, 4, 8, 16, and 32. Keep t-SNE/UMAP primarily for visualization initially; the current large reducer–clusterer sweep makes the inferential story harder to control.

Within each outer training set:

1. Independently cluster the same channels using different training-run subsets.
2. Compare channel memberships with ARI.
3. Measure held-out within-cluster coherence.
4. Record cluster sizes, between-cluster correlations, and effective dimensionality.

Do not select solely by highest stability: coarse partitions can be highly stable. Either report the predeclared resolution curve or select resolution using an additional training-only criterion, such as reconstruction of held-out channel activity.

Channels are the objects, while timepoints are their features. A centroid fitted across training timepoints cannot be directly applied to a different run with different timepoint features. For stability, independently recluster the same channels on another run and compare memberships. For downstream testing, **freeze the training-derived memberships and average those channels on the held-out run**.

When characterizing intact versus ablated responses, apply the same memberships—and a consistent training-fitted transformation—to all passes.

Replace the current “spread or stop” gate with separate conclusions:

- Are memberships reproducible?
- Do modality-preference differences reproduce?
- Does the compressed representation retain incremental AV prediction?

Failure of modality separation should stop that particular interpretation, not automatically stop compression or brain-alignment tests.

#### B2: Use anatomical brain targets first; make vertex clustering a secondary comparison

The vertex-cluster correlation floor near 0.41 is a description, not a reason by itself to reject those clusters. Genuine networks can share substantial stimulus-driven activity.

First test model representations against fixed anatomical parcels or vertices within ROIs. Then ask whether training-derived vertex clusters provide a better or simpler summary.

If retaining vertex clustering, fit it only on training runs, using native-TR data, and freeze memberships for held-out evaluation. Bin only when comparing to model representations. Use the same validation principle on both sides without requiring identical algorithms or numbers of clusters.

#### B3: Evaluate correspondence on unseen runs

The existing correlation matrix is symmetric under exchanging its two inputs. Reversing “model→brain” to “brain→model” just transposes the matrix; it is not independent replication.

Instead:

- Choose a model–brain cluster match on training data.
- Evaluate that fixed match on the held-out run.
- Compare selected matches with alternative matches.
- Test whether the model cluster’s modality preference predicts the brain target’s independently defined auditory–visual position.

Keep the full matrix as a descriptive output, but make held-out correspondence the result.

Fix the within-run shift null as planned. Use the same temporal shift across all model clusters within a run to preserve their dependence. Correct over the tested correspondence family. If any pairing or configuration selection uses the data being permuted, repeat that selection inside the null procedure.

#### B4: Test the content hypothesis directly once labels arrive

Use continuous annotation main effects and interactions to test whether anterior versus posterior regions differ in their relationship to speech–music versus speech–face combinations. Predictive interaction and coupling analyses should be separate endpoints.

This is where clustering could add interpretability: identify which stable model components carry the particular content-dependent effect, rather than trying to infer content from modality preference alone.

The general principle—fitting flexible representations on training data and evaluating generalization across stimuli, with subject generalization assessed separately—is consistent with the RSA inference framework already used in this project. See [Schütt et al., 2023](https://elifesciences.org/articles/82566).

### First practical deliverable

Build **one shared fold-aware evaluator** that accepts full embeddings, PCA scores, random projections, frozen cluster summaries, or within-cluster PC1 summaries and returns:

- baseline held-out \(R^2\) for A+V;
- extended held-out \(R^2\) for A+V+representation;
- incremental \(G=\Delta R^2\);
- per-run and per-clip held-out predictions and errors;
- ROI and vertex-level outputs;
- all training-fitted transformation and hyperparameter provenance.

Run it first on PE-AV and Nemotron L18. That answers the central additivity question while making the later scramble, compression, and clustering extensions directly comparable.

---

## Implementation status — 2026-09-07

### Implemented

- Added a shared nested evaluator with outer leave-one-run-out testing and inner run-wise regularization selection.
- Implemented the direct primary contrast, ordinary held-out \(R^2(A+V+J)-R^2(A+V)\), retaining negative values.
- Excluded the four repeated validation clips from the primary run-generalization analysis and retained them only for separate reliability work.
- Implemented training-only full, PCA, random-projection, cluster-mean, and within-cluster-PC1 representations. Transforms and their output scaling are fitted on training runs and saved for every fold.
- Implemented both compression questions: incremental AV at fixed full A+V, and matched-budget \(C_d(J)\) versus \(C_{d/2}(A)+C_{d/2}(V)\).
- Added anatomical/functional CIFTI ROI selection, vertex-level predictions, per-run and per-clip metrics, paired clip bootstrap intervals, sign-flip tests, errors, hyperparameters, and provenance.
- Separated representation random seeds from ridge-search random seeds so compression-seed comparisons are matched.
- Added a separate unimodal-model input for a future fold-confined scrambled-J analysis, so the correct intact A and V baseline can remain fixed; canonical global scramble models are rejected.
- Replaced the legacy global residual plus `r2_score_split` implementation with a direct fixed-clip A+V versus A+V+J comparison. The fixed-clip script is retained only as a compatibility/diagnostic analysis; the unseen-run evaluator is primary.
- Implemented independent-run channel reclustering, ARI, held-out frozen-membership coherence and reconstruction, effective dimension, seed stability, and a membership-randomization null that preserves cluster sizes.
- Added `incremental_av` and `channel_stability` modes to the existing encoding and clustering master runners. The encoding mode fails rather than silently falling back to a whole-cortex fit when the Glasser atlas is unavailable.
- Validated both main branches on real PE-AV/HCP group-average data in bilateral A1 using small smoke-test settings. These pilot values are software checks, not reportable estimates.

Implementation locations:

- `encoding/incremental_av.py`: data loading, ROI selection, outer-run evaluation, result caching, summaries, inference, and provenance.
- `encoding/shared/fold_evaluator.py`: grouped ridge fitting, run-wise inner CV, direct incremental-R² comparison, matched-budget compression comparison, clip metrics, bootstrap intervals, and sign-flip inference.
- `encoding/shared/compression.py`: training-fitted full, PCA, random-projection, cluster-mean, and within-cluster-PC1 transforms.
- `cluster/channel_stability.py`: independent-run memberships, ARI, randomized-membership nulls, frozen-membership held-out diagnostics, seed stability, and saved memberships.
- `encoding/variance_partition.py`: corrected fixed-clip compatibility analysis using direct A+V versus A+V+J prediction.
- `encoding/analysis.sh incremental_av`: domain-level execution entry point.
- `cluster/cluster.sh channel_stability`: domain-level execution entry point.
- `tests/test_fold_evaluator.py` and `tests/test_compression.py`: split-safety, scoring, compression, inference, and cluster-null tests.

### Remaining work that does not require content labels

- The factorial quartet interaction representation requires a crossed-pairing availability audit or targeted model extraction.
- Existing global scramble embeddings must not be used for the primary pairing test because their pairing crosses held-out boundaries.
- The provenance audit confirmed that both current scramble extractors permute globally and save no pairing indices. Depending on the held-out run, 111–117 training rows can contain held-run audio, with the same number of reverse crossings; 41–47 retained training rows can also consume audio from excluded validation clips. The encoding entry points now reject canonical `avscramble` models. Safe testing requires fold-specific joint re-extraction with separate training- and test-pool permutations and a saved row-level pairing manifest.
- Population inference still requires individual-subject fits and related-participant-aware resampling. Group-average results describe stimulus-locked alignment only.

### Segment content labels

Per-segment content labels are not available and are not expected. Every analysis that
required them is dropped, not deferred: content-conditioned coupling, animacy and
speech/music/face annotation tests, and the region-by-content interaction. The
label-free substitutes below replace them.

## Regional dissociation: anterior versus posterior cross-modal cortex

Both temporal cross-modal regions are nominally audiovisual. The claim under test is
that they are multimodal in different senses.

- `cca_a` — audio-dominant and **redundant**: video adds nothing beyond audio, and its
  representational richness is audio-category structure.
- `cca_p` — **synergistic**: video carries unique variance and the joint audiovisual
  representation exceeds the additive combination of its unimodal parts.

Unimodal controls: Glasser `A5` (auditory) and `FFC` (visual).

Two independent measurements, all label-free:

1. **Unique variance and synergy** (`encoding/roi_av_profile.py`). Per ROI, held-out
   `R2_A`, `R2_V`, `R2_additive`, `R2_joint`; `synergy = R2_joint - R2_additive`
   retaining sign; the participation ratio of the audio- and
   video-predicted ROI timecourses; and unique$_A$/unique$_V$ as genuine partial
   correlations between the ROI's observed timecourse and each band's out-of-fold
   prediction, controlling for the other band's out-of-fold prediction.

2. **Category structure without labels**, same script, second band configuration. The
   transcript text embedding stands in for audio-semantic category structure and the
   caption text embedding for visual-semantic structure. `cca_a` should be carried by
   transcript alone; `cca_p` should require both.

## Outstanding blockers

- (in progress) Cluster-mean and cluster-PC1 compression return held-out `R2` of -1 to -11 against an
  A+V baseline of +0.2497, systematically across folds, seeds, models and ROIs. The
  compression transforms themselves are clean; the fault is in the ridge fit. The
  compression comparison cannot be reported until this is resolved.
- (in progress) The held-out ROI alignment result is tested only against a circular-shift null, which
  any channel average beats. A size-matched random-partition control is required before
  the clusters can be said to carry structure.
- Population inference still requires individual-subject fits and related-participant-aware
  resampling; it is on hold until the group-average maps show promise.
