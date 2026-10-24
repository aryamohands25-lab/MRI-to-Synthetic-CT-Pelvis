
# MRI-to-Synthetic CT Generation for Pelvic Radiotherapy Planning Using Pix2Pix cGAN

**A Deep Learning Research Project in Medical Image Synthesis and MRI-Only Radiotherapy Planning**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange.svg)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-SynthRAD2023-purple.svg)](https://doi.org/10.5281/zenodo.7260705)
[![License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow.svg)]()

> **Research and educational use only. This model has not been clinically validated and must not be used for patient treatment or clinical radiotherapy dose calculation.**

## 1. Project Overview

Accurate radiotherapy planning requires information about tissue electron density, which is commonly derived from computed tomography (CT) images. Although magnetic resonance imaging (MRI) provides excellent soft-tissue contrast, its signal intensities do not directly correspond to electron density or Hounsfield Units (HU).

This project investigates the generation of **synthetic CT (sCT) images from pelvic MRI using a paired conditional Generative Adversarial Network (cGAN)** based on the Pix2Pix framework.

The system learns a mapping between spatially aligned pelvic MRI and CT slices using a U-Net generator and a PatchGAN discriminator. Its performance is evaluated on held-out patients using image-similarity metrics and region-specific HU error analysis.

The long-term motivation is to investigate computational approaches that could support MRI-only radiotherapy planning by reducing dependence on separately acquired CT scans and the associated inter-scan registration process.

### Key Highlights

- MRI-to-CT image translation using a Pix2Pix conditional GAN.
- Paired pelvic MRI and CT data from the SynthRAD2023 dataset.
- Patient-level training, validation, and test separation.
- Coverage-filtered preprocessing to address incomplete MRI coverage.
- Quantitative evaluation using MAE, PSNR, and SSIM.
- Region-specific analysis of air, soft tissue, and bone HU values.
- Reproducible Google Colab workflow with GPU-based training.

## 2. Research Objectives

The project has five primary objectives:

1. Develop an end-to-end deep learning pipeline for generating synthetic CT images from pelvic MRI.
2. Implement a Pix2Pix architecture that combines adversarial learning with pixel-wise reconstruction loss.
3. Evaluate generalisation to previously unseen patients using patient-level data separation.
4. Investigate how incomplete MRI coverage affects model evaluation and reconstruction quality.
5. Characterise region-specific HU errors, particularly the underestimation of bone density, which is important when considering future radiotherapy applications.

## 3. Dataset

**Dataset:** SynthRAD2023 Grand Challenge — Task 1, Pelvis

**Source:** [SynthRAD2023 dataset on Zenodo](https://doi.org/10.5281/zenodo.7260705)

The public dataset contains paired MRI and CT volumes, together with body masks, from two imaging centres.

| Dataset characteristic | Description |
|---|---|
| Selected patients | 48 |
| Centre A | 24 patients |
| Centre C | 24 patients |
| Training set | 28 patients |
| Validation set | 8 patients |
| Test set | 12 patients |
| Imaging modality | Pelvic MRI and CT |
| Model input | Single-channel axial MRI slice |
| Model target | Corresponding axial CT slice |
| Dataset format | NIfTI volumes |
| Evaluation unit | Held-out patients |

The dataset contains differences in MRI contrast and acquisition settings between centres. Centre A uses T1-weighted gradient-echo MRI, whereas Centre C uses T2-weighted spin-echo SPACE MRI.

These differences introduce domain variability and make cross-centre generalisation an important consideration.

**Data availability:** The dataset is not included in this repository. It is downloaded by the notebook from the public dataset source. Dataset usage and redistribution must comply with the applicable SynthRAD2023 terms, including CC BY-NC 4.0 as specified for this release.

## 4. Methodology

### 4.1 Pipeline Architecture

The workflow consists of data acquisition, preprocessing, model training, synthetic CT generation, and quantitative evaluation.

```text
SynthRAD2023 Pelvic MRI + CT
             |
             v
     Patient-Level Split
             |
             v
       Preprocessing
  - MRI intensity normalisation
  - CT HU clipping and scaling
  - Body-mask processing
  - Slice alignment and resizing
  - MRI coverage filtering
             |
             v
      Paired 2D Slices
             |
             v
       Pix2Pix cGAN
       /           \
      v             v
 U-Net Generator  PatchGAN
      |          Discriminator
      v             |
 Synthetic CT <-----+
      |
      v
 Inverse CT Intensity Scaling
      |
      v
 Evaluation on Held-Out Patients
      |
      +-- MAE
      +-- PSNR
      +-- SSIM
      +-- Region-Specific HU Error
```

### 4.2 Image Preprocessing

The preprocessing pipeline standardises MRI and CT images while preserving their spatial correspondence.

- **CT intensity processing:** CT values are clipped to the range of -1000 to 2000 HU and scaled to [-1, 1].
- **MRI intensity processing:** MRI intensities are clipped at the 99.5th percentile and normalised per patient to [-1, 1].
- **Body-mask processing:** Voxels outside the body mask are set to air, helping remove the CT couch and background.
- **Spatial processing:** Axial slices are padded to square dimensions and resized to 256 × 256 pixels.
- **Coverage filtering:** The v2 preprocessing pipeline removes slices in which less than 50% of the body has MRI signal, reducing the impact of incomplete MRI coverage.
- **Data separation:** Patients are assigned to training, validation, and test sets rather than splitting individual slices across these sets.

The coverage-filtering experiment compares the original preprocessing pipeline (v1) with the filtered pipeline (v2).

### 4.3 Pix2Pix Conditional GAN

The model consists of two neural networks trained in an adversarial framework.

**Generator — U-Net**

The generator receives an axial pelvic MRI slice and predicts the corresponding synthetic CT slice. Its encoder-decoder structure extracts hierarchical image features, while skip connections preserve spatial details and anatomical boundaries.

- Architecture: U-Net with skip connections
- Approximate parameter count: 54.4 million
- Input: Single-channel MRI
- Output: Single-channel synthetic CT

**Discriminator — PatchGAN**

The discriminator receives the paired MRI and real or generated CT images. It learns to distinguish real MRI–CT pairs from MRI–synthetic-CT pairs by assessing local image patches.

- Architecture: Conditional PatchGAN
- Approximate parameter count: 2.8 million
- Input: Concatenated MRI and CT channels

### 4.4 Training Configuration

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Input resolution | 256 × 256 |
| Optimiser | Adam |
| Learning rate | 0.0002 |
| Adam beta values | (0.5, 0.999) |
| Batch size | 16 |
| Maximum training duration | 40 epochs |
| Reconstruction loss | L1 loss |
| Adversarial loss | Binary cross-entropy (BCE) |
| L1 loss weight | 100 |
| Training precision | Mixed precision |
| GPU environment | Google Colab T4 |
| Model selection | Lowest validation MAE |

The generator objective combines adversarial loss and L1 reconstruction loss:

\[
\mathcal{L}_{G}
=
\mathcal{L}_{\mathrm{adv}}
+
100\mathcal{L}_{L1}
\]

The adversarial component encourages realistic image appearance, while the L1 component penalises voxel-wise intensity differences between generated and reference CT images.

The checkpoint with the lowest validation MAE is selected for evaluation.

## 5. Experimental Results

The final v2 model was evaluated on 12 held-out test patients. Metrics were calculated within the body mask and are reported as mean ± standard deviation across patients.

### 5.1 Overall Image Reconstruction Performance

| Experiment | MAE (HU) ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|---:|
| **Final model — v2, coverage-filtered training data** | **87.6 ± 10.1** | **24.44 ± 0.91** | 0.826 ± 0.027 |
| v1 model — all test slices | 91.0 ± 11.6 | 24.25 ± 0.92 | 0.832 ± 0.027 |
| v1 model — filtered test slices | 85.7 ± 9.6 | 24.47 ± 0.92 | 0.830 ± 0.027 |

**Interpretation:** The final model achieves a test MAE of 87.6 HU and a PSNR of 24.44 dB. The v1 model evaluated on filtered test slices achieves a slightly lower MAE of 85.7 HU. These results show that evaluation-set composition affects the measured error and that the v2 training experiment does not demonstrate a clear improvement over the v1 model on the filtered test slices.

The small test cohort and patient-to-patient variability mean that minor differences should not be interpreted as statistically significant without further analysis.

### 5.2 Cross-Centre Evaluation

| Centre | Test patients | MAE (HU) |
|---|---:|---:|
| Centre A | 6 | 90.7 ± 13.4 |
| Centre C | 6 | 84.6 ± 4.9 |

The observed MAE differs between the two centres. However, only six test patients represent each centre, so these results are insufficient to establish a reliable centre-specific performance difference.

### 5.3 Region-Specific HU Analysis

| Tissue region | HU range | MAE (HU) | Mean bias (HU) |
|---|---|---:|---:|
| Soft tissue | -100 to 200 HU | 55.9 | -11 |
| Air | Below -500 HU | 120.3 | +74 |
| Bone | Above 250 HU | 302.1 | -263 |

Region-specific analysis identifies bone reconstruction as the principal weakness of the current model.

The negative mean bone bias indicates systematic HU underestimation. The reported median bone HU is approximately 172 HU in the synthetic CT, compared with 424 HU in the reference CT.

This is an important limitation because accurate bone and tissue-density representation is relevant to radiotherapy dose calculation. Good overall image similarity does not establish adequate dosimetric accuracy.

### 5.4 Training and Validation Behaviour

| Evaluation | MAE (HU) |
|---|---:|
| Training slices | 26.3 |
| Validation set, selected checkpoint at epoch 38 | 97.7 |
| Constant-HU validation baseline | Approximately 333 |

The model substantially outperforms the constant-HU validation baseline. However, the large difference between training and validation errors indicates a generalisation gap.

Validation error plateaus after approximately 20 epochs, suggesting that simply extending training is unlikely to resolve the limitations associated with the small training cohort and the complexity of MRI-to-CT translation.

### 5.5 Qualitative Results

The repository is designed to include training curves, qualitative image comparisons, and failure-case visualisations.

| Visualisation | Purpose |
|---|---|
| `training_curves.png` | Tracks training and validation performance across epochs. |
| `qualitative_panel.png` | Supports visual comparison of MRI, reference CT, and synthetic CT. |
| `failure_cases.png` | Highlights challenging reconstructions and incomplete MRI coverage. |

These figures should be generated by the notebook and committed to the repository before they are linked or embedded in the final GitHub page.

## 6. Ablation and Coverage Analysis

The v1 and v2 experiments investigate the influence of MRI coverage filtering.

- The number of available slices decreases from 5,286 to 5,163 after filtering.
- Applying the coverage filter to the v1 test evaluation reduces MAE by approximately 5 HU.
- Training with filtered data does not show a clear improvement over the v1 model on the same filtered test slices.
- Scoring only regions where MRI signal is present reduces the reported MAE from 87.6 to 76.3 HU for the analysed v2 model.

The last comparison requires caution: restricting evaluation to MRI-covered regions also excludes the outer body rim. Therefore, the reduction in error cannot be attributed exclusively to improved MRI coverage.

## 7. Repository Structure

The notebook currently contains the end-to-end experimental workflow. For a clean and professional GitHub repository, the following structure is recommended:

```text
MRI-to-Synthetic-CT-Pelvis/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── MRI_to_CT_pix2pix_pelvis.ipynb
├── config/
│   ├── split.json
│   └── selected_patients.json
└── results/
    ├── figures/
    │   ├── training_curves.png
    │   ├── qualitative_panel.png
    │   └── failure_cases.png
    └── tables/
        ├── test_metrics.csv
        ├── test_metrics_v2.csv
        └── region_hu_summary_v2.csv
```

**Note:** This is the intended repository layout. Include only files that actually exist in your repository, and update the notebook link if you retain a different filename.

## 8. Installation and Execution

### Prerequisites

- A GitHub account and access to the repository.
- A Google account with access to Google Colab and Google Drive.
- A T4 GPU runtime, or another compatible GPU environment.
- Access to the SynthRAD2023 dataset under its applicable usage terms.

### Run in Google Colab

**Step 1 — Open the notebook**

Open `notebooks/MRI_to_CT_pix2pix_pelvis.ipynb` in Google Colab.

[Open Google Colab](https://colab.research.google.com/)

**Step 2 — Configure the runtime**

Select **Runtime → Change runtime type → T4 GPU**, where available.

**Step 3 — Run the setup cells**

Execute the setup section to install dependencies, mount Google Drive, initialise the environment, and configure paths.

**Step 4 — Download and prepare the data**

Run the data-download section. The notebook downloads the selected MRI, CT, and body-mask files from the dataset source, checks the downloaded files, and creates a backup archive on Google Drive.

The initial data preparation takes approximately 40 minutes under the conditions reported for this project. Actual time depends on download speed and runtime availability.

**Step 5 — Run preprocessing**

Execute the preprocessing sections to generate the paired 256 × 256 slices, perform the coverage filtering, and prepare the training, validation, and test datasets.

**Step 6 — Load the trained checkpoint or retrain**

Training is disabled by default using `RETRAIN = False`. The notebook expects the saved checkpoint at:

```text
MyDrive/mri2ct_capstone/checkpoints_v2/best.pt
```

If the checkpoint is available, the notebook can load it for evaluation. If it is not available, download the matching checkpoint from the repository's Releases page if one has been published, or set `RETRAIN = True` to retrain the model.

The reported training time is approximately 25 minutes for 40 epochs on a Colab T4 under the original experimental conditions. Runtime may vary.

**Step 7 — Evaluate the model**

Run the remaining cells to generate synthetic CT predictions, calculate quantitative metrics, inspect tissue-specific HU errors, and produce visualisations.

### Reproducibility

The patient split is stored in `config/split.json` in the intended repository layout. Using the same split, preprocessing configuration, model checkpoint, and evaluation procedure helps reproduce the reported results.

For exact reproducibility, also preserve the random seeds, dependency versions, selected patient identifiers, and model checkpoint associated with the experiment.

## 9. Limitations

The current implementation is a research baseline rather than a clinically validated synthetic CT system.

1. **Limited cohort:** Only 48 patients are used, including 12 held-out test patients.
2. **Generalisation gap:** Training MAE is considerably lower than validation MAE, indicating limited generalisation to unseen patients.
3. **Bone HU underestimation:** Bone-region MAE is 302.1 HU, with a mean bias of -263 HU.
4. **MRI domain variability:** The two centres use different MRI contrasts and acquisition settings.
5. **Two-dimensional modelling:** Independent 2D slices do not explicitly enforce consistency between adjacent slices.
6. **Incomplete MRI coverage:** Some MRI slices cover only part of the anatomy visible in CT, affecting reconstruction and evaluation.
7. **Limited input information:** The generator uses a single MRI channel.
8. **Evaluation scope:** MAE, PSNR, SSIM, and region-specific HU errors measure image reconstruction, not clinical safety or treatment accuracy.
9. **No dosimetric validation:** No radiotherapy dose calculation or clinical workflow evaluation has been performed.
10. **Checkpoint selection:** The validation set is used to select the best epoch, so its reported performance is subject to model-selection bias. The test set is reserved for final evaluation.

## 10. Future Work

Potential extensions include:

- Expanding the training cohort and improving representation of different MRI acquisition protocols.
- Investigating bone-weighted reconstruction losses and perceptual objectives.
- Comparing L1-only training against adversarial training with different loss weights.
- Exploring 2.5D and 3D architectures to improve anatomical continuity.
- Improving robustness to incomplete MRI coverage and cross-centre domain shifts.
- Evaluating additional HU calibration and tissue-specific reconstruction strategies.
- Integrating the synthetic CT outputs into a validated dose-calculation pipeline for research evaluation.
- Conducting larger, independent, and clinically relevant validation studies.

## 11. References

1. Thummerer, A., van der Bijl, E., Galapon, A., Verhoeff, J. J. C., Langendijk, J. A., Both, S., van den Berg, C. A. T., & Maspero, M. (2023). *SynthRAD2023 Grand Challenge dataset: Generating synthetic CT for radiotherapy.* [Dataset DOI](https://doi.org/10.5281/zenodo.7260705).

2. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

3. Pang, H., Zhang, T., Wu, Y., Chen, S., Qian, W., Yao, Y., Ye, C., Monkam, P., & Qi, S. (2026). *Generative Models for Medical Image Creation and Translation: A Scoping Review.* Sensors, 26(3), 862. [https://doi.org/10.3390/s26030862](https://doi.org/10.3390/s26030862).

## 12. License and Data Usage

**Code:** MIT License, provided in `LICENSE`.

**Dataset and derived data:** The SynthRAD2023 dataset is subject to its applicable licence and usage restrictions, including CC BY-NC 4.0 as specified for the dataset release. Non-commercial restrictions and attribution requirements must be respected.

The code licence does not override the dataset licence. Before redistributing trained weights, processed data, or other derived artefacts, verify that their distribution complies with the dataset terms.

## 13. Author

**Anagha Jayarajan**  
M.Sc. Data Science and Bio AI  
Computational Biology | Medical Imaging | Generative AI

*Project type: Academic research / Generative AI capstone project.*

---

**Disclaimer:** This repository is intended for research and educational purposes. The generated synthetic CT images have not been validated for clinical use, and the reported image-similarity metrics do not establish dosimetric accuracy or patient safety.

