# Tissue-Structure Reconstruction for Cross-Centre Generalization in Histopathology


Code, results and notes for the B.E. major project **"Tissue-Structure Reconstruction for Cross-Centre
Histopathology: Comparison with Stain Augmentation and Foundation Models"**
(Project PID 23, Project Part-I, course 22CAC11).

---

## 1. Paper and project information

| | |
|---|---|
| **Title** | Tissue-Structure Reconstruction for Cross-Centre Histopathology: Comparison with Stain Augmentation and Foundation Models |
| **Authors** | Achanta Badrinath, Peesari Sathvik Reddy |
| **Institution** | Department of CSE (AI & ML), Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India |
| **Venue** | Prepared for submission to IIITCON 2026 (Track 1). Under double-blind review; do not circulate this repo to reviewers as is. |
| **Paper** | 6-page IEEE-format paper. The PDF will be added to `docs/` after the review period. |
| **Project phase** | Part I complete (this repo). Part II planned (see Section 11). |

---

## 2. TL;DR

- We pretrain three backbones (ResNet18, ViT-Small/16, Swin-Tiny) with a **self-supervised tissue-structure
  reconstruction** task: recover the spatial arrangement of nine shuffled, partly masked patches from a 3×3 window.
- Evaluated by **leave-one-centre-out (LOCO)** on Camelyon17-WILDS with a frozen linear probe and patient-level
  bootstrap CIs, against ImageNet / random init, an HED **stain-augmentation** baseline, and three frozen
  **pathology foundation models** (Kaiko, Phikon, Lunit).
- **Result:** every trained strategy reaches strong cross-centre AUROC (0.91-0.99). Stain augmentation and the
  foundation models score highest. Reconstruction pretraining lifts ViT-Small and Swin-Tiny over ImageNet
  initialization using only the target dataset's own centres, but not ResNet18.
- **Why:** probes show raw colour statistics alone identify the hospital with **87.5%** accuracy (chance 20%), so a
  colour/centre shortcut dominates. Recon. ViT-Small has the smallest average in-domain-to-external drop across
  three external datasets. Attribution patterns depend on architecture.

---

## 3. Motivation

Histopathology models trained at one hospital often fail at another because of differences in staining, scanner
hardware and patient population. The recent response has been scale (foundation models pretrained on millions of
slides), which needs far more compute and data than a small academic study has. We ask a narrower question: can a
**task-specific self-supervised objective that retains tissue-architectural context** compete with stain augmentation
and full-scale foundation models, and *why* does each representation generalize (or not)?

We build on HistoLite (jigsaw-style pretext learning for histopathology domain generalization) and the original
jigsaw pretext task. Our task reconstructs the spatial **arrangement** of neighbouring patches, not pixel content.

### Contributions

1. A jigsaw-style tissue-structure reconstruction task assessed on three backbone families (CNN, ViT, Swin) under
   LOCO with patient-level bootstrap CIs.
2. A common-protocol comparison against a stain-augmentation baseline and three foundation models.
3. A probe battery (centre, position, patient identity, raw colour statistics, centre signature) showing raw colour
   alone recovers hospital identity at 87.5%.
4. Fixed-probe cross-dataset transfer to PatchCamelyon, NCT-CRC-HE-100K and LC25000 for seven models.
5. A quantitative explainable-AI analysis (Grad-CAM, attention rollout, centre-mass ratio) across eight models.

---

## 4. Method

```
 3x3 window of 9 patches (shuffled, up to a few masked)
        |
   per-patch backbone  (ResNet18 | ViT-Small/16 | Swin-Tiny, ImageNet-initialised)
        |
   projection + LayerNorm  -->  feature-level masking (learned mask token)
        |
   + learned slot embedding
        |
   Transformer context module (self-attention across the 9 slots)
        |            |
  permutation head   position head
   (500 classes)      (9 classes)
        \             /
         L = L_perm + L_pos
```

- **Pretext task:** predict which of 500 fixed permutations was applied and the true grid position of each patch.
- **Masking:** applied at the feature level with a learned mask token, so position must be inferred relationally.
- **Pretraining data:** four of the five centres only (the held-out centre is never seen).
- **Downstream:** the backbone is frozen, features are extracted once, then `StandardScaler` + logistic regression.
  AUROC is reported with a patient-level bootstrap 95% CI.

### Models compared

| Group | Models |
|---|---|
| Reconstruction (ours) | Recon. ResNet18, Recon. ViT-Small/16, Recon. Swin-Tiny |
| Baselines | ImageNet init (frozen), Random init (frozen), Stain Augmentation (HED, end-to-end ResNet18) |
| Foundation models (frozen) | Kaiko ViT-S/16, Phikon ViT-B/16, Lunit ViT-S/16 |

---

## 5. Data

| Dataset | Used for | Source |
|---|---|---|
| **Camelyon17-WILDS** | Pretraining, LOCO, probes, XAI (455,954 patches, 5 centres, patient-disjoint) | Hugging Face `wltjr1007/Camelyon17-WILDS` |
| PatchCamelyon | Cross-dataset transfer (test split) | HF `1aurent/PatchCamelyon` |
| NCT-CRC-HE-100K | Cross-dataset transfer (the `CRC_VAL_HE_7K` split; `TUM` class = tumour) | HF `1aurent/NCT-CRC-HE` |
| LC25000 | Cross-dataset transfer (adenocarcinoma + squamous = malignant) | HF `1aurent/LC25000` |

- **Primary split:** train on centres {0, 1, 3, 4}, test on centre 2 (370,900 train / 85,054 test patches).
- **LOCO:** five folds, each centre held out in turn.
- Datasets download automatically from the Hugging Face Hub at run time. Nothing is stored in this repo.

---

## 6. Results (from the paper)

### Table I. Primary split (centre 2 held out), AUROC with patient-bootstrap 95% CI

| Model | AUROC | 95% CI |
|---|---|---|
| Kaiko ViT-S/16 | 0.9825 | [0.9635, 0.9891] |
| Stain Augmentation | 0.9816 | [0.9494, 0.9916] |
| Lunit ViT-S/16 | 0.9783 | [0.9620, 0.9933] |
| Recon. Swin-Tiny | 0.9615 | [0.9298, 0.9724] |
| Recon. ViT-Small | 0.9605 | [0.9136, 0.9742] |
| Phikon | 0.9508 | [0.8309, 0.9887] |
| ImageNet baseline | 0.9473 | [0.9208, 0.9596] |
| Recon. ResNet18 | 0.8848 | [0.7832, 0.9172] |
| Random-Init | 0.6393 | [0.5523, 0.7163] |

### Table II. Leave-one-centre-out, mean AUROC over 5 folds

| Model | Mean AUROC | Std (across centres) |
|---|---|---|
| Stain Augmentation | 0.9884 | 0.0047 |
| Kaiko ViT-S/16 | 0.9873 | 0.0043 |
| Phikon | 0.9818 | 0.0176 |
| Lunit ViT-S/16 | 0.9814 | 0.0116 |
| Recon. ViT-Small | 0.9656 | 0.0075 |
| Recon. Swin-Tiny | 0.9615 | 0.0063 |
| ImageNet baseline | 0.9528 | 0.0066 |
| Recon. ResNet18 | 0.9148 | 0.0206 |
| Random-Init | 0.7797 | 0.1823 |

Std is the spread across the five held-out centres, not across training seeds.

### Table III. Linear probe vs. trained linear head (primary AUROC)

| Model | Linear probe | Trained head (5 epochs) |
|---|---|---|
| ImageNet baseline | 0.9473 | 0.8806 |
| Recon. ResNet18 | 0.8848 | 0.7791 |
| Recon. Swin-Tiny | 0.9615 | 0.9549 |
| Recon. ViT-Small | 0.9605 | 0.9621 |

The backbone stays frozen in both variants; only the classifier is trained differently. The head's training loss is still
falling at epoch 5, so the ResNet-based drops reflect a small training budget, not a general result.

### Table IV. Probe battery: what frozen features encode

| Representation | Centre ID | Position | Patient ID (30-way) |
|---|---|---|---|
| Raw colour statistics (no model) | 87.5% | n/a | n/a |
| Recon. ResNet18 | 28.5% | 99.7% | 31.1% |
| Recon. ViT-Small | 44.0% | 99.2% | 42.9% |
| Recon. Swin-Tiny | 38.1% | 99.7% | 50.3% |
| *chance* | *20.0%* | *11.1%* | *3.3%* |

Additional probe results produced by `probe_battery_centre_position_patient` and not yet in the paper:
Stain Aug (centre 36.0%, position 13.1%, patient 95.1%), Kaiko (98.9%, 11.2%, 99.9%), Lunit (89.0%, 10.9%, 99.9%),
Phikon (97.2%, 11.6%, 99.99%). Foundation-model features identify the hospital far more readily than the
reconstruction features.

### Table V. Cross-dataset fixed-probe transfer, balanced accuracy (%)

| Model | In-domain | PCam | NCT-CRC | LC25000 |
|---|---|---|---|---|
| Recon. ResNet18 | 78.7 | 66.4 | 73.8 | 60.0 |
| Recon. ViT-Small | 80.2 | 72.9 | 77.6 | 76.8 |
| Recon. Swin-Tiny | 83.3 | 73.7 | 68.8 | 57.5 |
| Stain Augmentation | 99.3 | 78.6 | 75.4 | 69.5 |
| Kaiko ViT-S/16 | 87.7 | 75.2 | 63.3 | 60.7 |
| Lunit ViT-S/16 | 85.9 | 77.4 | 83.9 | 63.4 |
| Phikon | 87.0 | 82.2 | 51.2 | 91.8 |

The probe is fit once on Camelyon17-WILDS and never refit on a target. Recon. ViT-Small has the smallest **average**
drop (4.4 points) but not the smallest on every dataset.

### Table VI. Centre-mass ratio (attribution inside the label-defining 32x32 px region; chance ~ 0.11)

| Model | In-domain | Held-out centre |
|---|---|---|
| ImageNet baseline | 0.276 | 0.266 |
| Recon. ResNet18 | 0.200 | 0.226 |
| Stain Augmentation | 0.218 | 0.200 |
| Recon. ViT-Small | 0.152 | 0.156 |
| Kaiko ViT-S/16 | 0.134 | 0.134 |
| Phikon | 0.113 | 0.125 |
| Lunit ViT-S/16 | 0.111 | 0.112 |
| Recon. Swin-Tiny | 0.017 | 0.026 |

Up to 200 correctly classified tumour patches per model and split. Grad-CAM for CNN-style models, attention rollout for
ViTs and foundation models.

---

## 7. Key findings

1. **All strategies generalize well**, but stain augmentation and the foundation models are best on the primary benchmark.
2. **Reconstruction pretraining is architecture-dependent:** it helps ViT-Small and Swin-Tiny, not ResNet18.
3. **A colour shortcut dominates:** raw colour statistics identify the hospital at 87.5%.
4. **Near-ceiling position accuracy (>=99%)** for all three reconstruction backbones shows the pretext task was learned,
   but that alone does not guarantee downstream generalization.
5. **Recon. ViT-Small gives the most stable transfer** to external datasets.
6. **Attribution is architecture-dependent:** Recon. Swin-Tiny falls well below chance (about 0.02); the cause is not
   established by this analysis.

---

## 8. Running the code

- **Environment:** a single-GPU Kaggle (or Colab) session. Main packages: `torch`, `torchvision`, `timm`,
  `transformers`, `datasets`, `kagglehub`, `scikit-learn`, `scikit-image` (HED stain augmentation), `pandas`,
  `matplotlib`, `seaborn`, `Pillow`, `tqdm`.
- **Structure:** each notebook is a standalone script with a `Config` class at the top (paths, seeds, sample sizes).
- **Checkpoints and cached features** are passed between notebooks as Kaggle datasets (Kaggle input paths appear in each
  `Config`). They are not stored in this repo.
- **Outputs** go to `/kaggle/working/...`. Save the final CSV tables and figures with the notebooks you commit.
- **Compute notes:** pretraining, feature extraction and the Stain-Aug baseline are the heavy steps (five end-to-end
  ResNet18 trainings for the baseline). Everything after feature extraction works on cached features and is cheap.

### Training setup (as in the notebooks)

Common: 3x3 windows, 500 permutations, learned mask token and slot embedding, AdamW, peak LR 3e-4, weight decay 0.05,
warm-up then cosine schedule, mixed precision, early stopping on position accuracy.

| Backbone | Notebook version | Mask prob. | Max masked patches | Max epochs | Patience |
|---|---|---|---|---|---|
| ResNet18 | v3 | 0.20 | 3 | 60 | 12 |
| ViT-Small | v3 | 0.20 | 3 | 60 | 15 |
| Swin-Tiny | v2 | 0.15 | 2 | 60 | 12 |

Stain-Aug baseline: end-to-end ResNet18, HED-space augmentation, 10 epochs per fold, AdamW, LR 1e-4, cosine annealing.

---

## 9. Reproducibility notes

1. **Stain-Aug provenance.** Tables I and II use the archived `stain_aug_folds_*` runs (primary AUROC 0.9816). The probe,
   transfer and explainability notebooks use checkpoints from the later standalone run (0.9826). Training is not
   bit-identical across the two runs.
2. **No seed replicates.** The "multi-seed" LOCO step refits a deterministic logistic regression on cached features, so its
   seed-wise std is 0. Variation reported in the paper is across held-out centres.
3. **Recipes differ by backbone** (table above). Check each `Config` before comparing numbers or re-running.
4. **Input size.** ViT-Small and Swin-Tiny use 224 px in the analysis notebooks; ResNet18 uses 96 px.
5. **Single runs.** Probe-battery and transfer results are single-seed (seed 42) and have no confidence intervals.

---

## 10. Limitations

1. No independent training-seed replicates for LOCO.
2. XAI figures show one example per model and split; the centre-mass ratio itself uses up to 200 samples.
3. Asymmetric comparison: reconstruction backbones are pretrained on the target dataset, while foundation models are
   used frozen with external pretraining.
4. One primary dataset; the three external datasets give fixed-probe evidence, not a second full LOCO study.
5. The centre-mass ratio uses the central 32x32 px region as a coarse reference, not a segmentation mask.
6. The 3x3 context, 500 permutations and masking schedule were fixed, not sensitivity-tested.

---

## 11. Part II roadmap (planned)

| Track | Description | Status |
|---|---|---|
| Core extensions | Patient-bootstrap LOCO replicates, full probe battery on foundation models and Stain-Aug features, XAI beyond one sample per split | Planned |
| Adversarial debiasing | Gradient-reversal centre-classifier head during reconstruction pretraining, to suppress the colour/centre shortcut | Planned (primary new contribution) |
| Weight averaging | Cyclical weight averaging across checkpoints | Planned |
| Robustness index | PathoROB-style robustness index alongside the probes | Planned |
| Ablations | Masking probability, masked-patch cap, permutation count, debiasing weight (primary split only) | Planned |
| Multi-scale reconstruction | 2x2 / 3x3 / 4x4 windows with a cross-scale consistency loss | Stretch goal, time-boxed |

Part II work will be added in a separate `part2/` folder; Part I stays untouched.

---

## 12. Acknowledgements

Built on Camelyon17-WILDS, PatchCamelyon, NCT-CRC-HE-100K and LC25000, and on the open Kaiko, Phikon and Lunit
pathology foundation models.

## 13. License

Released under the MIT License. See [`LICENSE`](LICENSE).
