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
| `1_Codes/` | AlexNet, VGG16, and SE-AlexNet (three FC insertion locations) code, plus the training script and Grad-CAM scripts |
| `2_Results/` | Training curves, 136 prediction CSVs, Grad-CAM heatmaps and numeric data |
| `3_Ablation(Codes)/` `4_Ablation(Results)/` | Convolutional-layer SE ablation (`SE-Conv-L1~L4`, not in the paper) |
| `5_FittingCurveAnalysis/` | Copy of the psychometric-curve fitting code and its data/results (canonical code lives in a separate repo, see below) |
| `scripts/` | Validation script for the prediction archive |
| `docs/` | GitHub Pages project page |

## Reproduce

### Environment

```bash
pip install -r requirements.txt
```

### Train

The FC-layer SE models (the paper) use `1_Codes/train.py`. Before running it, copy `se_model.py` and `mydataset.py` from `3_Ablation(Codes)/` into the same directory, and set the AffectNet dataset and pretrained-weight paths (they are hard-coded absolute paths in the script). Then:

```bash
python 1_Codes/train.py
```

The convolutional-layer SE ablation (not in the paper) lives in `3_Ablation(Codes)/` and takes command-line arguments:

```bash
cd 3_Ablation(Codes)
python run_all.py                                            # run every condition
python train.py --se_pos 1 --reduction 16 --trans Face        # one condition
```

### Predict

Prediction writes one 21-row CSV of "happy" probabilities per condition. The archived outputs are in `2_Results/1_Predict21Datapoints/` (120 SE-AlexNet conditions + 16 baselines). Check the archive with:

```bash
python3 scripts/validate_prediction_archive.py
```

For the convolutional-layer ablation, run:

```bash
cd 3_Ablation(Codes)
python predict.py --se_pos 1 --reduction 16 --trans Face --mask Full
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

- Insertion location: `Location-1` early (before FC6), `Location-2` mid (between FC6 and FC7), `Location-3` late (between FC7 and FC8).
- Pretraining: `FaceBased` (VGGFace2), `ObjectBased` (ImageNet).
- Mask condition: `Full`, `E` (eyes), `N` (nose), `M` (mouth).
- Reduction ratio: 2 / 4 / 8 / 16 / 32.

> The paper reports FC-layer insertion only (`Location-1/2/3`). Convolutional-layer insertion (`SE-Conv-L1~L4`) is a separate ablation not in the paper; see `3_Ablation(Codes)/` and `4_Ablation(Results)/`.

## Availability

- Model weights: <https://huggingface.co/JiayuMBao/SE-AlexNet>
- Psychometric-curve fitting: <https://github.com/JyBmegan/PsychometricFittingCurve>
- Grad-CAM / ROI analysis: <https://github.com/JyBmegan/GradCAM-ROI-SaliencyMapDecoder>
- Project page: <https://jybmegan.github.io/SE-AlexNet/>

## Citation

<!-- Citation to be added -->
