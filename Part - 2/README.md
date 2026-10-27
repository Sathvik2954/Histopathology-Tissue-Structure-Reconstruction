# Part II: Adjacency-Aware Contrastive Pretraining and Cross-Centre Style Mixing

Part II asks whether a stronger self-supervised objective, applied as continued pretraining of existing backbones,
yields more hospital-robust frozen features on Camelyon17-WILDS. Two methods are compared under one frozen
linear-probe protocol, with three seeds (42, 43, 44), on six backbones: ResNet18, ViT-S/16, Swin-T and the pathology
foundation models Kaiko ViT-S/16, Lunit ViT-S/16 and Phikon ViT-B/16.

- **Method 1, adjacency-aware contrastive pretraining.** Spatial neighbours in a 3×3 tissue lattice are positives. Only
  nearby same-slide patches are masked from the negatives, so distant same-slide patches act as hard negatives. A matched
  SimCLR run with identical data, schedule and negatives (the _self_ control) isolates the contribution of adjacency.
- **Method 2, Cross-Centre Style Mixing (CCSM).** A SimCLR objective with MixStyle applied between samples from different
  hospitals, and optionally a Fourier amplitude swap across hospitals. Arms: `control`, `ms` (MixStyle), `fms`
  (MixStyle + Fourier), and `msrand` (random partners, ResNet18).

## Repository layout

| Folder                               | Contents                                                                                                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `01_Method1_Adjacency_Pretraining/`  | `1a_ResNet18_ViT-S_Swin-T/` and `1b_Foundation_Kaiko_Lunit_Phikon/`: one notebook per backbone and seed (18)                                                       |
| `02_Method2_CCSM_Pretraining/`       | One notebook per backbone, covering all three seeds (6)                                                                                                            |
| `03_Evaluation_Frozen_Linear_Probe/` | `3a_Method1_Adjacency/` and `3b_Method2_CCSM/`: primary-split, LOCO and centre-ID evaluation (8)                                                                   |
| `04_Ablations/`                      | ABL1 negative-masking radius, ABL2 HED augmentation (Kaiko ViT-S); ABL3 Fourier-only, ABL4 random partners, ABL5 combined, ABL6 label efficiency (Lunit ViT-S) (8) |
| `05_Analysis/`                       | ABL7 cross-dataset transfer, ABL8 probe battery (a–d), ABL9 explainability (a–b) (7)                                                                               |

## Evaluation protocol

- Frozen features, `StandardScaler` + logistic regression (C = 1, class-balanced).
- **Primary split:** train on centres {0, 1, 3, 4}, test on centre 2, with patient-level bootstrap confidence intervals.
- **LOCO:** five folds, each centre held out in turn; mean AUROC reported.
- **Centre-ID probe:** five-way patch-level linear probe (chance 20%).
- Pretraining uses unlabeled images from centres {0, 1, 3, 4}. In LOCO folds that hold out one of these centres, its
  unlabeled images are part of the pretraining pool; the centre-2 primary split is entirely unseen.
- Input size: 96 px for ResNet18, 224 px for all other backbones. Results are mean ± std over seeds 42/43/44.

## Results

### Table 1. LOCO mean AUROC

| Backbone        | Original | SimCLR self | Adjacency (Method 1) | Δ adjacency − self (points) | Best CCSM arm (Method 2) |
| --------------- | -------- | ----------- | -------------------- | --------------------------- | ------------------------ |
| ResNet18        | 0.9528   | 0.9649      | **0.9780** ± 0.0003  | +1.31                       | 0.9686 (FMS)             |
| ViT-S/16        | 0.9573   | 0.9723      | **0.9828** ± 0.0030  | +1.05                       | 0.9737 (MS)              |
| Swin-T          | 0.9687   | 0.9752      | **0.9839** ± 0.0009  | +0.87                       | 0.9682 (MS)              |
| Kaiko ViT-S/16  | 0.9873   | 0.9797      | **0.9928** ± 0.0006  | +1.31                       | 0.9821 (FMS)             |
| Lunit ViT-S/16  | 0.9814   | 0.9801      | **0.9900** ± 0.0012  | +0.98                       | 0.9869 (FMS)             |
| Phikon ViT-B/16 | 0.9815   | 0.9892      | **0.9935** ± 0.0004  | +0.43                       | 0.9812 (FMS)             |

Stain-augmentation baseline from Part I: 0.9884. On the centre-2 primary split, adjacency reaches 0.9924 (Kaiko),
0.9921 (Lunit) and 0.9896 (Phikon).

### Table 2. CCSM arms, LOCO mean AUROC

| Backbone        | Original | Control | MixStyle   | MixStyle + Fourier |
| --------------- | -------- | ------- | ---------- | ------------------ |
| ResNet18        | 0.9528   | 0.8678  | 0.9382     | **0.9686**         |
| ViT-S/16        | 0.9573   | 0.9640  | **0.9737** | 0.9720             |
| Swin-T          | 0.9687   | 0.9665  | **0.9682** | 0.9572             |
| Kaiko ViT-S/16  | 0.9873   | 0.9806  | 0.9810     | 0.9821             |
| Lunit ViT-S/16  | 0.9814   | 0.9829  | 0.9863     | **0.9869**         |
| Phikon ViT-B/16 | 0.9815   | 0.9754  | 0.9787     | 0.9812             |

### Table 3. Lunit ViT-S ablations, LOCO mean AUROC

| Variant                             | AUROC           |
| ----------------------------------- | --------------- |
| Adjacency (Method 1)                | **0.9900**      |
| Combined: adjacency + CCSM          | 0.9883          |
| CCSM, MixStyle + Fourier            | 0.9869          |
| CCSM, MixStyle with random partners | 0.9862 / 0.9858 |
| CCSM, Fourier amplitude swap only   | 0.9847          |

### Table 4. Label efficiency, Lunit ViT-S (AUROC on centre 2 versus fraction of labelled training patches)

| Features           | 1%         | 5%         | 10%        | 25%        | 100%       |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Original           | 0.9643     | 0.9787     | 0.9831     | 0.9792     | 0.9783     |
| SimCLR self        | 0.9796     | 0.9850     | 0.9856     | 0.9859     | 0.9859     |
| Adjacency          | **0.9881** | **0.9911** | **0.9916** | **0.9917** | **0.9921** |
| MixStyle           | 0.9862     | 0.9881     | 0.9884     | 0.9869     | 0.9854     |
| MixStyle + Fourier | 0.9827     | 0.9866     | 0.9865     | 0.9866     | 0.9876     |
| Combined           | 0.9865     | 0.9899     | 0.9904     | 0.9906     | 0.9908     |

### Table 5. Hospital-identity probe on window-mean features, accuracy % (logistic regression / MLP; chance 20)

| Backbone        | Original    | Control     | Self            | Adjacency       | MixStyle    | MixStyle + Fourier | Combined        |
| --------------- | ----------- | ----------- | --------------- | --------------- | ----------- | ------------------ | --------------- |
| ResNet18        | 83.4 / 78.5 | 59.5 / 61.2 | 36.1 / 27.3     | **25.5** / 23.0 | 62.1 / 64.9 | 62.1 / 57.5        | n/a             |
| ViT-S/16        | 97.6 / 96.9 | 92.8 / 85.9 | 72.7 / 57.9     | **71.5** / 58.5 | 81.9 / 66.8 | 86.1 / 73.2        | n/a             |
| Swin-T          | 85.4 / 66.4 | 90.4 / 75.3 | 74.5 / 58.6     | **66.7** / 57.1 | 86.3 / 69.1 | 87.4 / 74.0        | n/a             |
| Kaiko ViT-S/16  | 98.7 / 98.7 | 93.9 / 81.4 | 74.9 / 54.7     | **72.9** / 67.4 | 87.5 / 71.5 | 85.7 / 65.3        | n/a             |
| Lunit ViT-S/16  | 97.1 / 95.2 | 98.9 / 95.3 | 83.3 / 68.0     | 80.9 / 64.7     | 98.9 / 95.3 | 93.5 / 90.1        | **75.7** / 57.9 |
| Phikon ViT-B/16 | 98.7 / 99.2 | 97.8 / 97.2 | **85.1** / 77.7 | 86.1 / 79.2     | 97.4 / 97.3 | 88.4 / 88.1        | n/a             |

### Key findings

1. Adjacency-aware pretraining outperforms its matched SimCLR control on all six backbones in 3/3 seeds, and improves
   on the original weights of every backbone.
2. CCSM improves ResNet18, ViT-S and Lunit; its Fourier arm is most useful on ResNet18, where plain SimCLR collapses.
3. On Lunit, the combination of adjacency and CCSM scores below adjacency alone, at full labels and at every label fraction.
4. Adjacency is the most label-efficient variant: with 1% of the labels it exceeds the original features trained on all labels.
5. Cross-dataset transfer: CCSM with the Fourier arm improves PatchCamelyon accuracy on all six backbones; adjacency
   mainly raises in-domain accuracy.
6. Hospital decodability falls on every backbone after continued pretraining, and does not by itself predict accuracy.
7. Attribution maps (attention rollout, Grad-CAM, centre-mass ratio) are stable across methods.

## Checkpoints

Checkpoints are attached to Kaggle notebooks as input datasets; they are not stored in this repository.

| Kaggle dataset                                                                          | Contents                                                 |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `sethu2954/adjc-kaiko-checkpoints`, `adjc-lunit-checkpoints`, `adjc-phikon-checkpoints` | Method 1 checkpoints, foundation models                  |
| `peesarisathvikreddy/adjclv2-checkpoints`                                               | Method 1 and self checkpoints, ResNet18 / ViT-S / Swin-T |
| `badrinathachanta/ccsm-checkpoints-1`                                                   | CCSM checkpoints, ResNet18 / ViT-S / Swin-T              |
| `badrinathachanta/ccsm-checkpoints-2`                                                   | CCSM checkpoints, Kaiko / Lunit / Phikon                 |
| `badrinathachanta/ablation-combined-checkpoints`                                        | Combined adjacency + CCSM, Lunit ViT-S                   |

## Running the notebooks

- Each notebook is a standalone script for a single-GPU Kaggle session (GPU and Internet on), with a configuration block at the top.
- Notebooks are limited to 11 hours and resume from saved per-model results; attach the previous run's output as an input to continue.
- Run in the order of the folder numbers: pretraining (01, 02), evaluation (03), ablations (04), analysis (05).
