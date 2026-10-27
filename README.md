# Cross-Centre Generalization in Histopathology: From Tissue-Structure Reconstruction to Adjacency-Aware Contrastive Pretraining and Cross-Centre Style Mixing

Self-supervised representation learning for hospital-robust histopathology, evaluated on Camelyon17-WILDS with
leave-one-centre-out (LOCO) frozen linear probes. The work is presented in two parts. Part I introduces a
tissue-structure reconstruction pretext task and compares it with stain augmentation and pathology foundation models.
Part II develops two stronger objectives: adjacency-aware contrastive pretraining and Cross-Centre Style Mixing (CCSM).

|                  |                                                                                                                                                                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Authors**      | Achanta Badrinath, Peesari Sathvik Reddy                                                                                                                                                                                                                                                        |
| **Institution**  | Department of CSE (AI & ML), Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad, India                                                                                                                                                                                                |
| **Part I paper** | _Self-Supervised Tissue-Structure Reconstruction for Cross-Center Generalization in Histopathology: A Comparison with Stain Augmentation and Foundation Models_. Accepted at the 7th International Conference on Data Engineering and Communication Technology (ICDECT 2026), publishing phase. |

---

## Overview

|                | Part I                                                                                                                          | Part II                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Folder**     | [`Part - 1/`](Part%20-%201/README.md)                                                                                           | [`Part - 2/`](Part%20-%202/README.md)                                                                                |
| **Title**      | Self-Supervised Tissue-Structure Reconstruction                                                                                 | Adjacency-Aware Contrastive Pretraining and Cross-Centre Style Mixing                                                |
| **Approach**   | Jigsaw-style reconstruction of nine shuffled, partly masked patches from a 3×3 tissue window                                    | Continued pretraining with spatial-neighbour positives (Method 1) and cross-hospital style mixing (Method 2)         |
| **Backbones**  | ResNet18, ViT-Small/16, Swin-Tiny                                                                                               | ResNet18, ViT-S/16, Swin-T, Kaiko, Lunit, Phikon                                                                     |
| **Comparison** | ImageNet and random initialisation, HED stain augmentation, Kaiko, Phikon and Lunit foundation models                           | Matched SimCLR control, original weights, stain augmentation                                                         |
| **Evaluation** | Primary split, LOCO, probe battery, cross-dataset transfer, explainability                                                      | Three seeds, primary split, LOCO, ablations, label efficiency, probe battery, cross-dataset transfer, explainability |
| **Highlight**  | Reconstruction lifts ViT-Small and Swin-Tiny over ImageNet initialisation; raw colour statistics identify the hospital at 87.5% | Adjacency beats its matched control on all six backbones, reaching LOCO AUROC 0.9780 to 0.9935                       |

### Part II results at a glance (LOCO mean AUROC, three seeds)

| Backbone        | Original | SimCLR control | Adjacency-aware (Method 1) |
| --------------- | -------- | -------------- | -------------------------- |
| ResNet18        | 0.9528   | 0.9649         | **0.9780**                 |
| ViT-S/16        | 0.9573   | 0.9723         | **0.9828**                 |
| Swin-T          | 0.9687   | 0.9752         | **0.9839**                 |
| Kaiko ViT-S/16  | 0.9873   | 0.9797         | **0.9928**                 |
| Lunit ViT-S/16  | 0.9814   | 0.9801         | **0.9900**                 |
| Phikon ViT-B/16 | 0.9815   | 0.9892         | **0.9935**                 |

HED stain-augmentation baseline (Part I): 0.9884. Full tables are in the part READMEs.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── Part - 1/                                  Tissue-structure reconstruction
│   ├── README.md
│   ├── pretraining/
│   │   ├── pretrain_recon_resnet18.ipynb
│   │   ├── pretrain_recon_vit_small.ipynb
│   │   └── pretrain_recon_swin_tiny.ipynb
│   ├── main_evaluation/
│   │   ├── feature_extraction.ipynb
│   │   ├── linear_probe_primary_and_loco.ipynb
│   │   └── stain_augmentation_baseline.ipynb
│   ├── analysis/
│   │   ├── probe_battery_centre_position_patient.ipynb
│   │   ├── ablation_probe_vs_trained_head.ipynb
│   │   ├── cross_dataset_transfer.ipynb
│   │   └── explainability_cmr.ipynb
│   └── archive/
│       ├── stain_aug_folds_0-2.ipynb
│       └── stain_aug_folds_3-4_xai_v1.ipynb
└── Part - 2/                                  Adjacency-aware contrastive pretraining and CCSM
    ├── README.md
    ├── 01_Method1_Adjacency_Pretraining/
    │   ├── 1a_ResNet18_ViT-S_Swin-T/          9 notebooks (3 backbones x 3 seeds)
    │   └── 1b_Foundation_Kaiko_Lunit_Phikon/  9 notebooks (3 models x 3 seeds)
    ├── 02_Method2_CCSM_Pretraining/           6 notebooks (one per backbone)
    ├── 03_Evaluation_Frozen_Linear_Probe/
    │   ├── 3a_Method1_Adjacency/              4 notebooks
    │   └── 3b_Method2_CCSM/                   4 notebooks
    ├── 04_Ablations/
    │   ├── ABL1_Negative_Masking_Radius_Kaiko/
    │   ├── ABL2_HED_Augmentation_Kaiko/
    │   ├── ABL3-4_CCSM_Fourier-only_and_Random-Partners_Lunit/
    │   ├── ABL5_Combined_Adjacency+CCSM_Lunit/
    │   └── ABL6_Label_Efficiency_Lunit/
    └── 05_Analysis/
        ├── ABL7_Cross_Dataset_Transfer/
        ├── ABL8_Probe_Battery/
        └── ABL9_Explainability_XAI/
```

---

## Data

| Dataset              | Used for                                                                                            | Source                                    |
| -------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **Camelyon17-WILDS** | Pretraining, LOCO evaluation, probes, explainability (455,954 patches, 5 centres, patient-disjoint) | Hugging Face `wltjr1007/Camelyon17-WILDS` |
| PatchCamelyon        | Cross-dataset transfer                                                                              | Hugging Face `1aurent/PatchCamelyon`      |
| NCT-CRC-HE-100K      | Cross-dataset transfer (`CRC_VAL_HE_7K` split, `TUM` = tumour)                                      | Hugging Face `1aurent/NCT-CRC-HE`         |
| LC25000              | Cross-dataset transfer (adenocarcinoma and squamous = malignant)                                    | Hugging Face `1aurent/LC25000`            |

- **Primary split:** train on centres {0, 1, 3, 4}, test on centre 2 (370,900 train / 85,054 test patches).
- **LOCO:** five folds, each centre held out in turn.
- Datasets are downloaded from the Hugging Face Hub at run time and are not stored in this repository.

## Environment

All notebooks are standalone scripts for a single-GPU Kaggle (or Colab) session with a configuration block at the top.
Main packages: `torch`, `torchvision`, `timm`, `transformers`, `datasets`, `kagglehub`, `scikit-learn`, `scikit-image`,
`pandas`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`. Checkpoints and cached features are passed between notebooks as
Kaggle datasets; outputs are written to `/kaggle/working/`.

## Evaluation protocol

All models are evaluated with frozen features, `StandardScaler` and a class-balanced logistic regression. AUROC is the
main metric, reported on the primary split (centre 2 held out) with patient-level bootstrap confidence intervals, and as
the mean over five LOCO folds. Part II reports mean ± std over seeds 42, 43 and 44.

---

## Citation

If you use this work, please cite the Part I paper:

```bibtex
@inproceedings{badrinath2026reconstruction,
  title     = {Self-Supervised Tissue-Structure Reconstruction for Cross-Center Generalization in Histopathology:
               A Comparison with Stain Augmentation and Foundation Models},
  author    = {Badrinath, Achanta and Reddy, Peesari Sathvik},
  booktitle = {7th International Conference on Data Engineering and Communication Technology (ICDECT)},
  year      = {2026},
  note      = {Accepted, in publication}
}
```

## Acknowledgements

Built on Camelyon17-WILDS, PatchCamelyon, NCT-CRC-HE-100K and LC25000, and on the open Kaiko, Phikon and Lunit pathology
foundation models.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
