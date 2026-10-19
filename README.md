# SE-AlexNet: Squeeze-and-Excitation Algorithm for Efficient Face Recognition

<p align="center">
    <a href="https://jybmegan.github.io/SE-AlexNet/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
    <a href="./0_InputImages/"><img src="https://img.shields.io/badge/Dataset-Github-orange" alt="Dataset"></a>
    <a href="https://huggingface.co/JiayuMBao/SE-AlexNet"><img src="https://img.shields.io/badge/Model%20Weights-Hugging%20Face-yellow" alt="Model Weights"></a>
</p>

This repository contains the code and results for the paper *A squeeze-and-excitation computational model for efficient face cognition*. It inserts Squeeze-and-Excitation (SE) channel-recalibration modules into a classical CNN (AlexNet) and studies how compression ratio, insertion position, and pretraining background shape the match between model outputs and human facial-emotion judgments.

## Repository structure

| Path | Contents |
| :--- | :--- |
| `0_InputImages/` | 21 test faces per condition (`Full`, `E`, `N`, `M`) |
| `1_Codes/` | AlexNet, VGG16, and the three archived SE-AlexNet model-family definitions, plus the training and Grad-CAM scripts |
| `2_Results/` | Training curves and the 136 prediction CSVs used by the paper |
| `scripts/` | Validation and public-release audit scripts |
| `docs/` | GitHub Pages project page |

## Reproduce

### Environment

```bash
pip install -r requirements.txt
```

### Reproduce the reported tables and statistics

Retraining is not required to reproduce the paper's numerical results. The 136
archived prediction files in `2_Results/1_Predict21Datapoints/` are the inputs
to the public analysis repositories.

First validate the complete prediction archive:

```bash
python3 scripts/validate_prediction_archive.py
```

Then use the two companion repositories:

1. [PsychometricFittingCurve](https://github.com/JyBmegan/PsychometricFittingCurve)
   recomputes all PSE, fit-quality, RMSE, ΔPSE, ANOVA, and post-hoc tables.
2. [GradCAM-ROI-SaliencyMapDecoder](https://github.com/JyBmegan/GradCAM-ROI-SaliencyMapDecoder)
   recomputes the manuscript ROI statistics from the archived Grad-CAM values.

The released checkpoints, configuration files, and canonical model-loading code
are available in the [SE-AlexNet model repository on Hugging Face](https://huggingface.co/JiayuMBao/SE-AlexNet).

### Model-location definitions in the released code

For reproducibility, the location names below describe the tensors used by the
released code and checkpoints:

- `Location-1`: SE is applied to the 256-channel feature maps after the final
  convolutional block and before flattening / FC6.
- `Location-2`: SE is applied to the 4096-dimensional vector between FC6 and
  FC7. In the archived definition and released checkpoints, this block uses
  reduction 16; the `R2`--`R32` names are retained experimental labels.
- `Location-3`: SE is also applied to the 256-channel feature maps after the
  final convolutional block and before flattening / FC6. It is a separately
  trained model family, but not a third tensor insertion site in the released
  architecture.

The exact executable definitions are the source of truth for model loading.

### Train and predict

Full retraining requires the licensed AffectNet images and labels, which are not
redistributed here. The original training record is `1_Codes/train.py`; the
three archived model-family definitions are in `1_Codes/SEAlexNetLocation1/`,
`1_Codes/SEAlexNetLocation2/`, and `1_Codes/SEAlexNetLocation3/`. These archival
scripts retain local path placeholders and are provided for methodological
traceability. They are not required to reproduce the reported result tables.
For evaluation without retraining, use the released checkpoints, configuration
files, and loader on Hugging Face. Prediction writes one 21-row CSV of "happy"
probabilities per condition. The public archive in
`2_Results/1_Predict21Datapoints/` contains 120 SE-AlexNet conditions and 16
baselines. Check the archive with:

```bash
python3 scripts/validate_prediction_archive.py
```

### Psychometric-curve fitting

Fit the psychometric curves and compute PSE / RMSE / ANOVA. See <https://github.com/JyBmegan/PsychometricFittingCurve>.

### Grad-CAM / heatmap and ROI analysis

Grad-CAM visualization and ROI saliency analysis. See <https://github.com/JyBmegan/GradCAM-ROI-SaliencyMapDecoder>.

## Prediction-file naming

```
Location-{1,2,3}-{FaceBased,ObjectBased}-{Full,E,N,M}-{2,4,8,16,32}.csv
{AlexNet,VGG-16}-{FaceBased,ObjectBased}-{Full,E,N,M}-raw.csv
```

- Model family: `Location-1`, `Location-2`, or `Location-3`; see the released-code definitions above for their actual tensor insertion sites.
- Pretraining: `FaceBased` (VGGFace2), `ObjectBased` (ImageNet).
- Mask condition: `Full`, `E` (eyes), `N` (nose), `M` (mouth).
- Reduction ratio: 2 / 4 / 8 / 16 / 32.

## Availability

- Model weights: <https://huggingface.co/JiayuMBao/SE-AlexNet>
- Psychometric-curve fitting: <https://github.com/JyBmegan/PsychometricFittingCurve>
- Grad-CAM / ROI analysis: <https://github.com/JyBmegan/GradCAM-ROI-SaliencyMapDecoder>
- Project page: <https://jybmegan.github.io/SE-AlexNet/>

## Public-release scope

The public repository contains only the files needed to validate the prediction
archive and connect it to the two companion analyses. It intentionally excludes
licensed training images, checkpoints, manuscript and submission materials,
generated Grad-CAM image archives, duplicate analysis folders, and legacy
ablation outputs. Checkpoints and the canonical model loader are linked from
Hugging Face; current psychometric and ROI analysis live in their companion
repositories.

Before committing or publishing, run:

```bash
python3 scripts/audit_public_release.py
```

## Citation

<!-- Citation to be added -->
