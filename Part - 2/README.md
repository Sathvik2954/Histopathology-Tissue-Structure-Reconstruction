# Part II: Adjacency-aware contrastive pretraining and Cross-Centre Style Mixing

Code for **Project Part-II** (Camelyon17-WILDS, frozen linear probe). 47 Kaggle notebooks, one folder per stage.
Every notebook is a standalone script (GPU on, Internet on); seeds are **42 / 43 / 44** and results are reported
as mean +/- std over the three seeds.

| Folder | Contents |
|---|---|
| `01_Method1_Adjacency_Pretraining/` | Method 1: adjacency-aware contrastive pretraining, plus the matched SimCLR "self" control. `1a` ResNet18 / ViT-S / Swin-T, `1b` Kaiko / Lunit / Phikon. One notebook per backbone and seed. |
| `02_Method2_CCSM_Pretraining/` | Method 2: Cross-Centre Style Mixing (MixStyle across centres, optional Fourier amplitude swap) inside SimCLR. One notebook per backbone, all seeds. Arms: `control`, `ms`, `fms` (+ `msrand` on ResNet18). |
| `03_Evaluation_Frozen_Linear_Probe/` | Primary split (train centres 0,1,3,4 / test centre 2) and LOCO evaluation of the checkpoints, plus the centre-ID probe. `3a` Method 1, `3b` Method 2. `EVAL_M1_adjacency_earlier_version` is kept for reference only. |
| `04_Ablations/` | ABL1 negative-masking radius and ABL2 HED augmentation (Kaiko ViT-S, parts A/B); ABL3 Fourier-only and ABL4 random partners (Lunit); ABL5 combined adjacency + CCSM (Lunit); ABL6 label efficiency (Lunit). |
| `05_Analysis/` | ABL7 cross-dataset transfer (all six backbones), ABL8 probe battery (a-d), ABL9 explainability (a-b). |

## Protocol

- Frozen features, `StandardScaler` + logistic regression (C=1, class-balanced).
- **Primary:** train on centres {0,1,3,4}, test on centre 2 (patient-level bootstrap CI). **LOCO:** five folds, mean AUROC.
- Backbones: ResNet18 (96 px), ViT-S/16, Swin-T, Kaiko ViT-S/16, Lunit ViT-S/16, Phikon ViT-B/16 (224 px).
- Pretraining uses unlabeled images from centres {0,1,3,4} only.
- **Transductive caveat:** for LOCO folds that hold out centres 0, 1, 3 or 4, that centre's unlabeled images are in the
  pretraining pool. Only the centre-2 primary split is fully clean. The matched self control is the cleaner attribution.

## Main findings (LOCO mean AUROC, three seeds)

| Backbone | Original | SimCLR self | **Adjacency** | CCSM best arm |
|---|---|---|---|---|
| ResNet18 | 0.9528 | 0.9649 | **0.9780** | 0.9686 (FMS) |
| ViT-S | 0.9573 | 0.9723 | **0.9828** | 0.9737 (MS) |
| Swin-T | 0.9687 | 0.9752 | **0.9839** | 0.9682 (MS) |
| Kaiko | 0.9873 | 0.9797 | **0.9928** | 0.9821 (FMS) |
| Lunit | 0.9814 | 0.9801 | **0.9900** | 0.9869 (FMS) |
| Phikon | 0.9815 | 0.9892 | **0.9935** | 0.9812 (FMS) |

Part-I stain-augmentation baseline: 0.9884.

1. Adjacency beats its matched self control on all six backbones (3/3 seeds each, +0.43 to +1.31 points).
2. CCSM is backbone-dependent: it helps ResNet18, ViT-S and Lunit, and does not help Swin-T, Kaiko or Phikon.
   On ResNet18 the SimCLR control collapses (0.8678), so part of "CCSM vs control" is avoiding that collapse.
   The Fourier-only variant is weaker than MixStyle, and random partners are equivalent to cross-centre partners.
3. Combining adjacency and CCSM is not additive on Lunit (0.9883 vs 0.9900 for adjacency alone, 0/3 seeds above).
4. Label efficiency (Lunit): adjacency is best at every label fraction (1% to 100%), and with 1% of the labels it
   beats the original features trained on all labels.
5. Hospital identifiability does not predict accuracy: across and within backbones, lower centre decodability does not
   mean higher AUROC.
6. Explainability (centre-mass ratio): no evidence that either method moves attribution toward the label-defining region.
7. Cross-dataset: CCSM + Fourier improves PatchCamelyon transfer on all six backbones; adjacency mainly raises
   in-domain accuracy and does not consistently improve NCT-CRC-HE or LC25000 transfer.

## Checkpoint datasets (Kaggle inputs)

Datasets are attached as notebook inputs (`kagglehub` downloads do not work in non-interactive commit runs).

| Dataset | Contents |
|---|---|
| `sethu2954/adjc-{kaiko,lunit,phikon}-checkpoints` | Method 1 checkpoints for the foundation models |
| `peesarisathvikreddy/adjclv2-checkpoints` | Method 1 + self checkpoints for ResNet18 / ViT-S / Swin-T |
| `badrinathachanta/ccsm-checkpoints-1` | CCSM checkpoints, ResNet18 / ViT-S / Swin-T |
| `badrinathachanta/ccsm-checkpoints-2` | CCSM checkpoints, Kaiko / Lunit / Phikon |
| `badrinathachanta/ablation-combined-checkpoints` | Combined adjacency + CCSM (Lunit) |

Long notebooks stop at 11 h and resume from per-model JSON results, so a re-run continues where the last one ended.

## Notes

- Each notebook is one code cell with a config block at the top.
- `04_Ablations/ABL1`, `ABL2` are split into parts A/B (seeds) to fit the Kaggle session limit.
- Results are not stored in this repo; outputs are written to `/kaggle/working/...`.
