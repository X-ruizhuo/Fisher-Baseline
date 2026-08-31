# Historical Fisher-Guided Bi-C2R: Research Design

## 1. Tentative Title

English:

> Historical Fisher-Guided Parameter Consolidation for Re-indexing-Free Lifelong Person Re-identification

Chinese:

> 历史 Fisher 信息引导的无重索引终身行人重识别参数巩固方法

Tentative method name:

> HFPC-BiC2R: Historical Fisher-guided Parameter Consolidation for Bi-C2R

The method name is provisional. The paper should avoid presenting standard EWC as the sole novelty. The final positioning should emphasize the cooperation between historical parameter importance and compatible representation learning under the RFL-ReID protocol.

## 2. Research Problem

Re-indexing-Free Lifelong Person Re-identification (RFL-ReID) sequentially learns from datasets arriving at different stages. Historical gallery images cannot be revisited or re-indexed after a model update. Consequently, the final query features extracted by the updated model must retrieve historical gallery features produced or updated across earlier stages.

Bi-C2R addresses this problem mainly in feature space:

- BiCT-Net transfers features between old and new representation spaces.
- BiCD preserves feature alignment and inter-instance relationships.
- BiAD distills identity discrimination in both transfer directions.
- DFF adaptively fuses model parameters and historical gallery features.

However, the backbone remains optimized on every new dataset without an explicit record of which parameters are important to historical identity knowledge. Repeated updates can therefore move historically important parameters, causing:

1. degradation of old-domain identity discrimination;
2. accumulated drift of the embedding geometry;
3. increasing difficulty in transferring historical gallery features;
4. a growing mismatch between final query features and repeatedly updated gallery features.

The central problem addressed in this work is:

> How can a compatible representation framework preserve historically important model parameters without retaining or revisiting historical images, while maintaining sufficient plasticity for new domains?

## 3. Core Hypothesis

At the end of stage t, the current dataset D_t is still available and the final deployed model theta_t has already learned that stage. The empirical Fisher information estimated using D_t and theta_t provides a parameter-wise approximation of the importance of theta_t to the identity discrimination learned at that stage.

Saving this Fisher statistic, rather than historical images, allows stage t+1 to selectively penalize changes to historically important parameters:

```text
historical feature compatibility + historical parameter stability
              -> improved long-term RFL-ReID
```

The method therefore uses:

```text
D_t + final deployed model theta_t -> Fisher F_t
```

The saved F_t is used as old Fisher at stage t+1. Current-stage data are not passed through the previous model to construct a proxy Fisher.

## 4. Design Decisions

The first implementation adopts the following fixed decisions:

1. Use the previous-stage historical Fisher, estimated from the previous-stage dataset and final previous-stage model.
2. Use diagonal empirical Fisher rather than a full or low-rank matrix.
3. Estimate Fisher from identity cross-entropy only.
4. Keep the Bi-C2R knowledge-distillation temperature fixed at T = 2.
5. Compute Fisher after DFF parameter fusion, so Fisher corresponds to the actual deployed model.
6. Accumulate Fisher online to preserve more than the most recent stage.
7. Apply Fisher constraints to the shared ReID model, not to the bidirectional transfer networks.
8. Exclude newly added classifier rows from the historical penalty.
9. Do not store or revisit historical images after the corresponding stage ends.

## 5. Method Overview

At stage t, the method has:

- current data D_t;
- previous deployed model theta_(t-1);
- saved historical Fisher F_(t-1);
- current trainable model theta initialized from theta_(t-1);
- forward and backward BiCT networks.

The stage contains four steps.

### Step 1: Initialize the Current Stage

Copy the previous deployed model as the frozen teacher and parameter anchor:

```text
theta_old <- theta_(t-1)
theta     <- theta_(t-1)
```

Expand the identity classifier from C_old classes to C_old + C_t classes. Preserve the previous classifier rows and initialize new rows from current identity centers, following the existing baseline.

### Step 2: Joint Compatible Training

Train on D_t with the original Bi-C2R objectives plus historical Fisher consolidation:

```text
L_total = L_BiC2R + lambda_fim * L_fim
```

The old model continues to process current-stage images for BiCD, BiAD, affinity preservation, and bidirectional feature transfer. This branch is separate from Fisher estimation.

### Step 3: Dynamic Feature Fusion

After current-stage training, calculate the original Bi-C2R adaptive coefficient alpha_t from the difference between old-model and new-model affinity matrices. Fuse the old and current model parameters using the original DFF procedure:

```text
theta_t = alpha_t * theta_trained + (1 - alpha_t) * theta_(t-1)
```

Update historical gallery features through forward BiCT transformation and the same DFF coefficient.

### Step 4: Estimate and Save Current Fisher

Using the final fused model theta_t and current-stage data D_t, estimate the current Fisher contribution F_new,t. Merge it with the saved historical Fisher and save:

```text
checkpoint_t = {
    model: theta_t,
    fisher: F_t,
    old_parameters: theta_t,
    stage: t
}
```

After this step, D_t does not need to be retained for future training.

## 6. Fisher Estimation

### 6.1 Probabilistic Definition

For parameter theta_i, the diagonal Fisher is approximated by:

```text
F_new,t,i = (1 / N_t) * sum_n [d log p(y_n | x_n, theta_t) / d theta_i]^2
```

Since identity cross-entropy is:

```text
L_id = -log p(y | x, theta)
```

the sign disappears after squaring the gradient. In practice:

```text
F_new,t,i = average[(d L_id / d theta_i)^2]
```

### 6.2 Fisher-Generating Loss

The main method uses only identity cross-entropy:

```text
L_fisher_source = L_id
```

Triplet loss is excluded from the main Fisher definition because it does not directly define a conditional label likelihood and its gradient depends strongly on batch composition and hard-sample mining.

The full Bi-C2R objective must not be used to estimate Fisher because it would mix parameter importance with:

- transfer-network optimization;
- old-model distillation;
- manually scaled compatibility losses;
- domain-specific transfer difficulty;
- the Fisher regularizer itself.

In particular, using L_fim to estimate Fisher would create a circular definition.

### 6.3 Estimation Loader

Use the current stage `init_loader` or an equivalent deterministic loader:

- resize and normalize only;
- no random erasing;
- no random crop;
- no horizontal flip;
- no sample replay.

This improves reproducibility and reduces augmentation-induced variance in Fisher estimates.

### 6.4 Estimation Budget

The initial implementation should support a configurable number of Fisher batches:

```text
fisher_num_batches = all | 50 | 100
```

The default research setting should use the complete `init_loader` when feasible. A batch-budget ablation can later assess whether a cheaper estimate is sufficient.

## 7. Online Historical Fisher

Using only the most recent dataset Fisher would bias protection toward the latest stage. Therefore, maintain an online historical Fisher.

For shared parameters:

```text
F_t = gamma * F_(t-1) + Normalize(F_new,t)
```

The initial default is:

```text
gamma = 1.0
```

### 7.1 Normalization

Fisher magnitude can differ substantially across layers and stages. Use tensor-wise mean normalization:

```text
Normalize(F_i) = F_i / (mean(F_i) + epsilon)
```

with:

```text
epsilon = 1e-12
```

For numerical robustness, optionally clamp extreme normalized values:

```text
F_i <- clamp(F_i, 0, fisher_clip)
```

The default can leave clipping disabled. Clipping should be introduced only if logs show a small number of parameters dominating L_fim.

### 7.2 Why Online Accumulation Is Needed

At stage 5, a Fisher estimated only on stage 4 cannot explicitly preserve the importance learned from stages 1-3. Online accumulation carries historical importance forward without storing a separate Fisher matrix per stage.

Storage remains approximately one additional tensor per protected parameter, independent of the number of stages.

## 8. Historical Fisher Consolidation Loss

At stage t > 1:

```text
L_fim = 1/2 * sum_i F_(t-1),i * (theta_i - theta_old,i)^2
```

where:

- theta_old is the final deployed parameter from stage t-1;
- theta is the current trainable parameter;
- F_(t-1) is the online historical Fisher saved at stage t-1.

The complete objective is:

```text
L_total =
    L_id
  + L_triplet
  + L_affinity
  + L_bca
  + L_bcr
  + L_bad
  + L_bdc
  + lambda_fim * L_fim
```

The symbols correspond to the existing Bi-C2R base, compatible alignment, compatible relationship, anti-forgetting distillation, and directional consistency objectives.

## 9. Protected Parameter Scope

### 9.1 Main Setting

Protect:

1. ResNet-50 backbone convolution parameters;
2. trainable normalization scale parameters in the ReID model;
3. BN-neck trainable parameters;
4. previous identity classifier rows.

Do not protect:

1. newly added classifier rows;
2. forward BiCT network parameters;
3. backward BiCT network parameters.

### 9.2 Rationale

The backbone and BN neck carry historical identity representation. Previous classifier rows correspond to historical identities. New classifier rows must remain plastic. BiCT networks are stage-specific adaptation mechanisms and require freedom to learn the current domain transformation.

Constraining BiCT with historical Fisher would also require defining which historical transfer task its parameters should preserve, which is outside the initial contribution.

## 10. Classifier Expansion and Fisher Alignment

At stage t, the classifier changes from shape:

```text
[C_old, D] -> [C_old + C_t, D]
```

Historical Fisher has shape `[C_old, D]`. The Fisher penalty applies only to the shared prefix:

```text
L_fim_cls =
    1/2 * sum F_old_cls *
    (W_current[:C_old] - W_old)^2
```

No historical penalty is applied to:

```text
W_current[C_old:]
```

When estimating F_new,t after stage t:

- retain accumulated Fisher for previous classifier rows;
- estimate and save Fisher for current identity rows;
- do not overwrite old-row importance with gradients where old classes appear only as negatives for current samples.

Classifier update rule:

```text
F_cls,t[:C_old] = gamma * F_cls,t-1
F_cls,t[C_old:] = Normalize(F_new,t,current_rows)
```

This avoids treating current-stage negative-class gradients as direct evidence of old-class importance.

## 11. Relationship to Existing Bi-C2R Modules

| Component | Space | Responsibility |
|---|---|---|
| BiCT-Net | Feature transformation | Transfers old and new representations bidirectionally |
| BiCD | Feature and relation space | Preserves compatibility and relational structure |
| BiAD | Discriminative output space | Distills old and new identity discrimination |
| DFF | Parameter and gallery feature fusion | Adapts fusion to adjacent-domain discrepancy |
| Historical Fisher consolidation | Parameter space | Protects historically important ReID parameters |

The proposed component is complementary rather than substitutive:

```text
Bi-C2R controls what representations should remain compatible.
Historical Fisher controls which parameters should resist excessive movement.
```

## 12. Distillation Temperature Decision

Keep the baseline knowledge-distillation temperature fixed at:

```text
T = 2
```

Do not integrate the NSC adaptive-temperature mechanism in the main method because:

1. current Fisher is available only after current-stage training;
2. estimating current Fisher during training would introduce substantial overhead;
3. using current data through the old model would contradict the selected historical-Fisher definition;
4. Bi-C2R already contains DFF-based domain discrepancy adaptation;
5. changing Fisher consolidation and temperature simultaneously would weaken causal attribution.

Fixed-temperature sensitivity can be evaluated as an auxiliary experiment with T in `{1, 2, 4, 8}`.

## 13. Stage-Level Algorithm

```text
Input:
    sequential datasets D_1 ... D_T
    Bi-C2R model and transfer networks
    lambda_fim, gamma

Initialize:
    fisher_old = None
    deployed_model = ImageNet-initialized ReID model

For stage t = 1 ... T:
    old_model = deepcopy(deployed_model)
    old_parameters = copy(old_model parameters)

    expand classifier for identities in D_t

    for each training batch from D_t:
        compute original Bi-C2R loss

        if t > 1:
            compute L_fim using fisher_old and old_parameters
            L_total = L_BiC2R + lambda_fim * L_fim
        else:
            L_total = L_BiC2R

        update current ReID model and BiCT networks

    compute DFF coefficient alpha_t
    fuse current model with old_model
    update historical gallery features
    deployed_model = fused model

    estimate F_new,t using D_t, deployed_model, and identity CE
    fisher_old = online_update(fisher_old, F_new,t, gamma)

    save deployed_model and fisher_old
```

## 14. Integration Points in the Baseline

The implementation should remain narrowly scoped.

### `continual_train.py`

Responsibilities to add:

- maintain `fisher_old` and `old_parameters` across stages;
- pass them into `Trainer`;
- estimate Fisher after `linear_combination`;
- save Fisher in or next to stage checkpoints;
- restore Fisher during resume/evaluation workflows.

### `reid/trainer.py`

Responsibilities to add:

- compute historical Fisher penalty during current-stage training;
- log raw and weighted `L_fim`;
- support classifier prefix matching;
- avoid applying Fisher to transfer networks.

### New Utility Module

Recommended file:

```text
reid/utils/fisher.py
```

Responsibilities:

- select protected parameters;
- estimate diagonal Fisher;
- normalize Fisher tensors;
- merge online Fisher;
- compute Fisher penalty;
- align expanded classifier tensors;
- serialize and deserialize Fisher state.

This isolates the new mechanism from the already large training entry point.

## 15. Configuration

Recommended initial arguments:

```text
--use-fisher
--fisher-weight
--fisher-gamma
--fisher-num-batches
--fisher-normalize
--fisher-protect-classifier
```

Initial defaults for experiments:

```text
use_fisher = true
fisher_weight = selected from {1e-6, 1e-5, 1e-4, 1e-3}
fisher_gamma = 1.0
fisher_num_batches = all
fisher_normalize = tensor_mean
fisher_protect_classifier = true
temperature = 2
```

The correct `fisher_weight` must be selected from observed loss scales rather than copied from CAF, because diagonal normalized Fisher and the Bi-C2R loss magnitudes differ from the reference method.

## 16. Experimental Protocol

### 16.1 Dataset Orders

Use the existing Bi-C2R orders as the primary benchmark:

```text
Order 1:
Market1501 -> CUHK-SYSU -> DukeMTMC -> MSMT17 -> CUHK03

Order 2:
DukeMTMC -> MSMT17 -> Market1501 -> CUHK-SYSU -> CUHK03
```

Keep the same identity count, image size, sampler, epochs, optimizer, and random seed as the reproduced Bi-C2R baseline.

### 16.2 Main Comparisons

| Method | Compatible learning | Parameter protection |
|---|---:|---:|
| Fine-tuning/Base | No or baseline base | None |
| Bi-C2R | Yes | None |
| Bi-C2R + Uniform L2 | Yes | Uniform parameter penalty |
| Bi-C2R + Recent Fisher | Yes | Most recent stage Fisher only |
| Bi-C2R + Online Historical Fisher | Yes | Proposed cumulative Fisher |

Uniform L2 is essential. It demonstrates whether gains come from Fisher-based importance rather than merely reducing parameter movement.

### 16.3 Primary Metrics

Report for both L-ReID and RFL-ReID:

- mean Average Precision (mAP);
- Rank-1 accuracy;
- average performance across learned datasets;
- final-stage performance on each dataset;
- Average Forgetting (AF) for mAP and Rank-1.

### 16.4 Additional Diagnostics

Measure:

1. parameter drift per layer:

```text
||theta_t,l - theta_(t-1),l||_2
```

2. Fisher-weighted parameter drift:

```text
sum_i F_old,i * (theta_t,i - theta_old,i)^2
```

3. old-domain embedding drift, when evaluation images are available:

```text
1 - cosine(f_t(x), f_(t-1)(x))
```

4. historical gallery transformation error or query-gallery compatibility gap;
5. current-stage performance to quantify loss of plasticity;
6. Fisher estimation time and checkpoint storage overhead.

## 17. Ablation Studies

### 17.1 Fisher Source Loss

| Variant | Importance source |
|---|---|
| CE Fisher | Identity cross-entropy, main method |
| Triplet importance | Triplet loss only |
| Hybrid importance | CE + beta * Triplet |
| Full-loss importance | Complete Bi-C2R objective |

The main paper should call only the CE variant a standard empirical Fisher. Other variants should be described as task-aware gradient importance.

### 17.2 Historical Scope

| Variant | Update rule |
|---|---|
| Recent Fisher | F_t = F_new,t |
| Decayed Online | F_t = gamma F_(t-1) + F_new,t |
| Full Online | gamma = 1 |

Test at least:

```text
gamma in {0.5, 0.8, 1.0}
```

### 17.3 Protected Parameter Scope

Compare:

1. backbone only;
2. backbone + BN neck;
3. backbone + BN neck + old classifier rows, proposed;
4. all ReID model parameters;
5. ReID model + BiCT networks.

This experiment should demonstrate that selective protection preserves plasticity better than constraining all components.

### 17.4 Fisher Weight

Search logarithmically:

```text
lambda_fim in {1e-6, 1e-5, 1e-4, 1e-3}
```

If normalized Fisher makes these values too weak, expand the range based on measured weighted-loss magnitude. Select the value on a validation protocol without tuning separately for each dataset order.

### 17.5 Fisher Normalization

Compare:

- no normalization;
- global mean normalization;
- tensor-wise mean normalization, proposed;
- optional percentile clipping.

### 17.6 Fisher Estimation Budget

Compare:

```text
10, 50, 100, all batches
```

This assesses the effectiveness-efficiency trade-off.

### 17.7 Fixed Temperature Sensitivity

Keep this outside the main contribution:

```text
T in {1, 2, 4, 8}
```

It verifies that gains do not depend on an accidental temperature choice while preserving a controlled study.

## 18. Expected Results and Success Criteria

The method is considered promising if it achieves all of the following:

1. improves final average RFL-ReID mAP and Rank-1 over reproduced Bi-C2R;
2. reduces AF on early datasets, especially Market1501 in Order 1;
3. outperforms uniform L2 parameter regularization;
4. does not cause a large decline on the current-stage dataset;
5. provides consistent improvements across both dataset orders and multiple seeds;
6. adds acceptable Fisher estimation and storage overhead.

A useful initial threshold is:

```text
average RFL-ReID gain >= 0.8 mAP
AF reduction >= 1.0 point
current-stage drop <= 0.5 mAP
```

These are research targets, not guaranteed outcomes.

## 19. Failure Modes and Mitigations

### Excessive Stability, Weak Plasticity

Symptom: old datasets improve but the current dataset declines substantially.

Mitigation:

- reduce `lambda_fim`;
- protect backbone only;
- use gamma < 1;
- exclude later backbone blocks from protection as an ablation.

### Fisher Scale Dominates Training

Symptom: weighted L_fim is much larger than the original Bi-C2R loss.

Mitigation:

- tensor-wise normalization;
- log raw and weighted L_fim;
- use logarithmic weight search;
- optionally clip extreme Fisher values.

### No Improvement over Uniform L2

Interpretation: the Fisher estimate does not identify useful historical importance beyond generic parameter anchoring.

Mitigation or analysis:

- inspect layer-wise Fisher distributions;
- compare CE Fisher with hybrid task-aware importance;
- verify that Fisher is estimated after DFF;
- verify classifier offset labels and gradient accumulation;
- measure correlation between Fisher magnitude and actual forgetting after parameter perturbation.

### Improvement Only on One Order

Interpretation: the method may be sensitive to domain order or protect recent knowledge too strongly.

Mitigation:

- test decayed online Fisher;
- report per-stage and per-domain analysis;
- avoid claiming general robustness until multiple orders support it.

## 20. Paper Positioning

### 20.1 Main Problem Statement

Existing RFL-ReID compatible-learning methods primarily preserve historical knowledge by aligning old and new representations. However, they do not explicitly characterize which parameters in the continually updated ReID model encode historical identity knowledge. As training proceeds across heterogeneous domains, unrestricted parameter drift can weaken old-domain discrimination and amplify the error of repeated gallery-feature updates.

### 20.2 Method Statement

We introduce a historical Fisher-guided parameter consolidation mechanism into Bi-C2R. At the end of each stage, the method estimates diagonal empirical Fisher information from the current data and final deployed model, and carries it into subsequent stages as a compact historical parameter-importance memory. During later updates, historically important parameters receive stronger constraints, while newly introduced classifier parameters and compatible transfer networks remain plastic.

### 20.3 Distinction from CAF/NSC

The proposed method should be differentiated carefully:

- CAF/NSC estimates Fisher under its own continual distillation design and additionally uses Fisher evolution for adaptive temperature control.
- The proposed method fixes the distillation temperature and focuses on historical-stage Fisher as an explicit memory of learned identity importance.
- Fisher is computed after Bi-C2R dynamic model fusion so that the importance corresponds to the actual deployed model.
- Online Fisher accumulation and classifier-aware alignment are designed for the expanding identity classifier and multi-stage Bi-C2R pipeline.

This distinction must be validated experimentally. Merely adding standard EWC to Bi-C2R is not sufficient as a strong paper contribution.

## 21. Innovation Points

### Innovation 1: Historical Identity Parameter Memory

We propose a historical Fisher-based parameter memory for RFL-ReID. Unlike feature-only knowledge preservation, it records which parameters of the final deployed model are important to identity discrimination at each stage. This information is retained as compact statistics rather than historical images.

Suggested paper wording:

> We introduce a historical Fisher-guided parameter memory that explicitly characterizes the contribution of model parameters to previously learned identity discrimination, enabling selective old-knowledge preservation without retaining historical images.

### Innovation 2: Parameter-Feature Cooperative Compatibility

We integrate historical parameter consolidation with bidirectional compatible representation learning. Fisher regularization stabilizes historically important ReID parameters, while BiCT, BiCD, BiAD, and DFF preserve cross-model feature compatibility and update historical gallery features.

Suggested paper wording:

> We establish a parameter-feature cooperative continual learning framework, where historical Fisher consolidation suppresses destructive parameter drift and bidirectional compatible learning maintains retrieval compatibility between updated queries and historical gallery features.

### Innovation 3: Deployment-Aligned Online Fisher Consolidation

Fisher is estimated after dynamic model fusion and accumulated online. It therefore corresponds to the model actually deployed at the end of each stage and carries multi-stage importance forward with constant-stage storage complexity. Classifier-aware alignment preserves historical rows while leaving new identity rows unconstrained.

Suggested paper wording:

> We develop a deployment-aligned online Fisher update strategy tailored to the expanding classifier of lifelong ReID, preserving accumulated historical importance while maintaining plasticity for newly introduced identities.

## 22. Contribution Paragraph Draft

> First, we identify a parameter-stability limitation in existing re-indexing-free compatible representation learning: although old and new features are explicitly aligned, historically important parameters remain vulnerable to destructive drift during sequential domain updates. Second, we propose a historical Fisher-guided parameter consolidation mechanism that estimates identity-related parameter importance from each stage's final deployed model and preserves it without storing historical images. Third, we integrate the proposed parameter memory with bidirectional compatible representation learning, yielding a parameter-feature cooperative framework that jointly improves historical discrimination and cross-model retrieval compatibility. Finally, we design an online Fisher accumulation and classifier-aware alignment strategy for multi-stage identity expansion, and evaluate its effectiveness through extensive comparisons, ablations, forgetting analysis, and efficiency measurements.

## 23. Abstract-Level Draft

> Re-indexing-free lifelong person re-identification requires a model to continuously learn from heterogeneous data while retrieving historical gallery features without revisiting their original images. Existing compatible representation methods mainly mitigate incompatibility by aligning old and new feature spaces, but leave the parameter drift of the continually updated ReID model insufficiently constrained. Such drift progressively weakens historical identity discrimination and increases the difficulty of repeated gallery-feature transfer. To address this issue, we propose a historical Fisher-guided parameter consolidation framework built upon bidirectional compatible representation learning. At the end of each stage, diagonal empirical Fisher information is estimated using the current data and final deployed model to construct a compact historical parameter-importance memory. During subsequent stages, changes to historically important parameters are selectively penalized, while new classifier parameters and compatible transfer networks remain plastic. We further introduce an online Fisher accumulation and classifier-aware alignment strategy to preserve multi-stage knowledge under expanding identity spaces. Together with bidirectional feature transfer, compatible distillation, and dynamic feature fusion, the proposed approach provides cooperative protection in both parameter and feature spaces without retaining historical images.

## 24. Claims the Paper Should Avoid

Do not claim:

- that diagonal Fisher exactly represents all historical knowledge;
- that the method preserves privacy in a formal cryptographic sense;
- that Fisher regularization alone solves feature incompatibility;
- that the method is the first use of Fisher information in continual learning;
- that the method is fundamentally different from EWC without demonstrating task-specific design and evidence;
- that improvements generalize beyond RFL-ReID without experiments.

Use the more defensible description:

> Fisher provides a parameter-importance approximation for historical identity knowledge and complements feature-space compatible learning.

## 25. Minimum Experimental Evidence for a Paper

Before treating the method as a viable paper contribution, require:

1. reproduced Bi-C2R results under at least two orders;
2. at least three random seeds for the main comparison;
3. superiority over uniform L2 anchoring;
4. superiority of online Fisher over recent-only Fisher;
5. parameter-scope ablation;
6. Fisher-weight sensitivity;
7. old/new performance trade-off analysis;
8. parameter drift or Fisher-weighted drift visualization;
9. Fisher computation and storage overhead;
10. a clear comparison and discussion relative to CAF/NSC.

## 26. Final Recommended Version

The approved initial research method is:

```text
Bi-C2R
+ previous-stage historical diagonal empirical Fisher
+ CE-based Fisher estimation
+ post-DFF Fisher estimation
+ online Fisher accumulation
+ backbone, BN-neck, and old-classifier protection
+ new-classifier and BiCT plasticity
+ fixed distillation temperature T = 2
```

Its main scientific question is:

> Can deployment-aligned historical parameter importance complement bidirectional feature compatibility and reduce long-term forgetting in RFL-ReID without retaining historical images?

