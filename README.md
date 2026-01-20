# SE-AlexNet: Enhancing Face Perception via Dimension Reduction


## 1. Experimental Conditions & Variables

This study investigates the impact of Dimension Reduction on facial emotion recognition through Squeeze-and-Excitation (SE) modules.

### Summary of Conditions

| Variable | Description | Details / Values |
| :--- | :--- | :--- |
| **Model Architectures** | 5 distinct architectures | 1. **AlexNet** (Classic Benchmark)<br>2. **VGG16** (Classic Benchmark)<br>3. **SE-AlexNet Early** (Code: `SEAlexNetLocation1`)<br>4. **SE-AlexNet Mid** (Code: `SEAlexNetLocation2`)<br>5. **SE-AlexNet Late** (Code: `SEAlexNetLocation3`) - *Best Performance* |
| **Reduction Ratio** | Compression level in SE modules | $r = \{2, 4, 8, 16, 32\}$<br>*(Applicable only to SE-AlexNet variants)* |
| **Pre-training Basis** | Source of feature learning | **Face-Based** (Pre-trained on VGGFace2)<br>**Object-Based** (Pre-trained on ImageNet) |
| **Stimuli Types** | Input data conditions | **Full** (Full Face), **M** (Masked/Bubbles), **E** (Expression specific), **N** (Neutral) |


## 2. Repository Structure

The file structure is organized by the experimental data flow: **Inputs -> Codes -> Raw Results -> Analysis Data**.

```text
.
├── 0_InputImages/          # Dataset used for training and psychophysical testing
│   ├── E/                  # Expression images
│   ├── Full/               # Full face stimuli
│   ├── M/                  # Masked stimuli (Bubbles)
│   └── N/                  # Neutral/Other condition
│
├── 1_Codes/                # PyTorch Implementations
│   ├── AlexNet_Classic/    # Standard AlexNet (Benchmark)
│   ├── VGG16/              # Standard VGG16 (Benchmark)
│   ├── SEAlexNetLocation1/ # "Early" Integration (SE modules at lower layers)
│   ├── SEAlexNetLocation2/ # "Mid" Integration (SE modules at middle layers)
│   ├── SEAlexNetLocation3/ # "Late" Integration (SE modules at FC layers)
│   ├── train.py            # Main training script
│   └── predict.py          # Inference script for generating probability CSVs
│
├── 2_Results/              # Generated Outputs
│   ├── 0_Metrics/          # Training Logs (Accuracy & Loss curves .jpg)
│   ├── 1_Predict21Datapoints/ # Raw prediction CSVs (Model probabilities)
│   ├── 2_GradCAMHeatMap/   # Visualization images (Grad-CAM results)
│   └── 3_GradCAMDatapoints/ # Numerical data for Heatmaps (.txt)
│
├── 3_Ablation(Codes)/              # Unified Python Scripts for parametric study
│   ├── se_model.py                 # Dynamic model definition (Supports SE_Pos 1-5 & Squeeze Ratios)
│   ├── train.py                    # Training script (Configurable for different conditions)
│   ├── predict.py                  # Inference script (Matches training config)
│   ├── gradcam.py                  # Generates both Heatmaps (.png) and Raw Values (.csv)
│   ├── misc_functions.py           # Helper functions for visualization & image processing
│   └── mydataset.py                # Custom DataLoader for AffectNet
│
├── 4_Ablation(Results)/            # Experiment-Centric Output Structure
│   └── SeC1_FaceBased_squeeze32/   # [Example Condition Folder]
│       ├── acc.jpg                 # Training Accuracy Curve
│       ├── loss.jpg                # Training Loss Curve
│       ├── traindata.csv           # Epoch-wise training metrics
│       ├── Predict/                # Inference Outputs
│       │   └── SeC1_FaceBased_squeeze32_Full.csv
│       ├── datapoint/              # Grad-CAM Raw Numerical Data
│       │   ├── Full_1.csv          # Pixel-level attention matrix (224x224)
│       │   ├── Full_2.csv
│       │   └── ...
│       └── gradcam/                # Grad-CAM Visual Heatmaps
│           ├── Full_1.png          # Heatmap overlay image
│           ├── Full_2.png
│           └── ...
│
├── CollectionOfROIAnalysis.pptx    # Summary presentation of Region of Interest analysis
└── README.md

```
## 3. Psychometric Curve Analysis

The MATLAB code used to fit the psychometric functions (Experiment 1 & 2 behavioral analysis), calculate Points of Subjective Equality (PSE), and compare thresholds is hosted in a separate repository.

**Analysis Repository:** [**PsychometricFittingCurve**](https://github.com/JyBmegan/PsychometricFittingCurve)

*Note: The `3_Code4FitCurve` folder in this repository primarily contains the sorted input data (`.csv`) and output figures required for the analysis pipeline.*


## 4. Usage Guidelines

### Model Training
The training script allows selecting the model architecture, reduction ratio, and pre-training basis.

### Visualization (Grad-CAM)
To generate the heatmaps found in `2_Results/2_GradCAMHeatMap`, use the `gradcam.py` script located within each specific model folder (e.g., `1_Codes/SEAlexNetLocation3/gradcam.py`).

### Data Analysis Flow
1.  **Train Models**: Use `1_Codes/train.py`.
2.  **Generate Predictions**: Use `1_Codes/predict.py` to create raw CSVs in `2_Results/1_Predict21Datapoints`.
3.  **Format for MATLAB**: Data is organized into `3_Code4FitCurve/Data`.
4.  **Curve Fitting**: Run the MATLAB scripts (linked above) to generate the final psychometric curves found in `3_Code4FitCurve/Results`.

## 5. Ablation Study & Updated Structure

This section outlines the structure for the detailed Ablation Study. Unlike the separate folders in Section 2, this part uses a unified code structure where model architecture (SE Location) and parameters (Squeeze Ratio) are controlled dynamically.

### Key Components

1. ```se_model.py```: Now accepts arguments to insert SE blocks at any layer (1-4) with varying reduction ratios.

2. ```gradcam.py```: Updated to output both visual PNGs and raw CSV data for precise ROI analysis.

3.  Results are now grouped by Experimental Condition (e.g., SeC1...) rather than by file type. This ensures that the Model Weights, Logs, Predictions, and Grad-CAM data for a specific experiment are stored together.