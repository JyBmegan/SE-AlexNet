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
│   ├── run_all.py                  # Automated batch processing script
│   ├── se_model.py                 # Dynamic model definition
│   ├── train.py                    # Training script (supports argparse)
│   ├── predict.py                  # Inference script (supports argparse)
│   ├── gradcam.py                  # Visualization script (supports argparse)
│   ├── misc_functions.py           # Helper functions
│   └── mydataset.py                # Custom DataLoader
│
├── 4_Ablation(Results)/            # Experiment-Centric Output Structure
│   └── SeC1_FaceBased_squeeze32/   # [Example Condition Folder]
│       ├── current_model_arch.csv  # Detailed layer-by-layer model structure
│       ├── acc.jpg                 # Training Accuracy Curve
│       ├── loss.jpg                # Training Loss Curve
│       ├── traindata.csv           # Epoch-wise training metrics
│       ├── predict/                # Inference Outputs
│       │   ├── SeC1_FaceBased_squeeze32_Full.csv
│       │   ├── SeC1_FaceBased_squeeze32_E.csv
│       │   ├── SeC1_FaceBased_squeeze32_M.csv
│       │   └── SeC1_FaceBased_squeeze32_N.csv
│       ├── datapoint/              # Grad-CAM Raw Numerical Data
│       │   ├── SeC1-FaceBased-squeeze32-Full
│       │   │   ├── Full_1.csv
│       │   │   ├── Full_2.csv
│       │   │   ├── ......
│       │   │   └── Full_21.csv
│       │   ├── SeC1-FaceBased-squeeze32-E
│       │   ├── SeC1-FaceBased-squeeze32-M
│       │   └── SeC1-FaceBased-squeeze32-N
│       └── gradcam/                # Grad-CAM Visual Heatmaps
│           ├── SeC1-FaceBased-squeeze32-Full
│           │   ├── Full_1.png
│           │   ├── Full_2.png
│           │   ├── ......
│           │   └── Full_21.png
│           ├── SeC1-FaceBased-squeeze32-E
│           ├── SeC1-FaceBased-squeeze32-M
│           └── SeC1-FaceBased-squeeze32-N
│
├── CollectionOfROIAnalysis.pptx    # Summary presentation of Region of Interest analysis
├── CheckList4Train&Predict&GradCAM.xlsx
├── README.md
└── 5_FittingCurveAnalysis
    ├── Coding
    │      FitPsycheCurveWH.m
    │      PsychometricFittingCurve.m
    ├── Data
    └── Results
            Final_Analysis_Results.xlsx
            Human_Fitting_Parameters.xlsx
            Location-1-FaceBased-E.fig
            Location-1-FaceBased-E.m

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

## 6. Ablation Study Workflow & Usage

This section details the step-by-step workflow for the ablation experiments (located in `3_Ablation(Codes)`). Please refer to the provided `CheckList4Train&Predict&GradCAM.xlsx` for the complete list of experimental conditions.

### Step 1: Model Configuration

Before training, you must manually set the reduction ratio in the model definition.

1.  Open `se_model.py`.
2.  Locate `class SEBlock`.
3.  Modify the `reduction` parameter to your desired condition: `2, 4, 8, 16, 32`.

```python
# Inside se_model.py
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=32): # <--- Modify this value (e.g., 32)
        ...
```
### Step 2: Training

1. Open `train.py`.

2. Go to the `main` function (approx. line 125).

3. Locate the code block marked with comments `check/check point`.

4. Modify the following variables to define your experiment:

```python
# Inside train.py
current_se_pos = 1       # Insert location (1, 2, 3, 4, or 5)
squeeze_ratio = 32       # Must match the value in se_model.py
trans_con = 'Face'       # 'Face' or 'Object'
trans_path = 'face'      # 'face' or 'object'
...
output_dir = ...         # Ensure the pre-trained weight path is correct
```

**Important:** Absolute Paths Since the output path is absolute, please verify and update:

* `dataset_dir` and `csv_path` for train_dataset and validate_dataset.

* If running on a different device, update the root directory paths.

**Output:** Upon completion, a folder named after the condition (e.g., `SeC1_FaceBased_squeeze32`) will be created containing:

**Weight Path**: https://drive.google.com/drive/folders/1-zN6m67eHEvylYtCxSZ1m8g-8IsMF7L_?usp=drive_link

* AlexNet.pth (Trained Weights)

* traindata.csv (Training Logs)

* acc.jpg & loss.jpg (Performance Curves)

### Step 3: Inference (Prediction)

1. Open `predict.py`.

2. Locate the parameter settings at the beginning of `main`.

3. Ensure `current_se_pos`, `squeeze_ratio`, and `trans_con` exactly match the trained model settings (otherwise, weights will fail to load).

4. Modify `mask_con` to generate predictions for different stimuli types.

```python
# Inside predict.py
current_se_pos = 1       # Keep consistent with training
squeeze_ratio = 32       # Keep consistent with training
trans_con = 'FaceBased'  # Keep consistent with training

# Variable Condition:
mask_con = 'Full'        # Options:
                         # 'Full' (No Mask)
                         # 'E' (Eyes Masked)
                         # 'N' (Nose Masked)
                         # 'M' (Mouth Masked)
```
**Important** You MUST ensure the para in se_model and predict are SAME!  If you are training one condition, and you want to predict or visualize another condition, pay attention to this point.

**Output:**

A folder named `Predict` will be created inside the experiment folder.

It generates a CSV file compatible with the [PsychometricFittingCurve Repository](https://github.com/JyBmegan/PsychometricFittingCurve) for behavioral analysis.

### Step 4: Visualization (Grad-CAM)

1. Open `gradcam.py`.

2. Go to the `main` function (approx. line 122).

3. Locate the parameters marked with check point.

4. Set the parameters identical to the `predict.py` step (ensure matching Training conditions).

```python
# Inside gradcam.py
# Example for generating heatmaps for 'Eyes Masked' condition
current_se_pos = 1       # Keep consistent with training
squeeze_ratio = 32       # Keep consistent with training
trans_con = 'Face'       # Keep consistent with training 
mask_con = 'E'           # Target stimuli type (Full, E, N, M)
```

**Output:** 

Two folders will be created inside the experiment folder:

* **gradcam:** Contains the visual heatmaps (Attention Maps).

* **datapoint:** Contains the raw numerical data for the heatmaps (CSVs).

### [New Feature] Model Architecture Logging

During the training process (Step 2), the script now automatically generates a file named current_model_arch.csv inside the result folder. This file records the exact layer configuration, parameter count, and structural details of the generated model.

Example Content (`current_model_arch.csv`):

```python
Layer Name,Layer Type,Parameters,Configuration
features,Sequential,2469952,Sequential(...)
features.0,Conv2d,23296,"Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4)...)"
features.3,SEBlock,256,SEBlock(
features.3.fc1,Linear,128,"Linear(in_features=64, out_features=2, bias=False)"
...
classifier.6,Linear,32776,"Linear(in_features=4096, out_features=8, bias=True)"
```
## 7. Automated Batch Processing

To streamline the ablation study, a `run_all.py` script is provided. This script automatically iterates through all experimental conditions **(Pre-training type, SE Position, Reduction Ratio)** and executes the full pipeline **(Train $\to$ Predict $\to$ GradCAM)** for **all 4 mask conditions**.

### How to Use:

1. Open a terminal in the 3_Ablation(Codes) directory.

2. Run the script:

```python
python run_all.py
```

### Workflow Logic

The script operates as follows:

1. **Iterates Conditions:** Loops through `trans_con` (Face/Object), `se_pos` (1-4), and `ratios` (32, 16, 8, 4, 2).

2. **Skip Mechanism:** It checks a `completed_task`` list inside the script. If a specific condition (e.g., Face, Pos 1, Ratio 8) is already finished, it skips it to save time.

3. **Execution:**

    * Runs `train.py` with the current parameters.

    * If training is successful, it immediately runs `predict.py` for all 4 masks (Full, E, N, M).

    * Finally, it runs `gradcam.py` for all 4 masks.

### Switching Between Auto and Manual Modes

* **Auto Mode (Default):** The scripts (`train.py`, `predict.py`, `gradcam.py`) are now configured to accept command-line arguments (e.g., `--se_pos 1 --reduction 32`). `run_all.py` utilizes this interface.

* **Manual Mode:** If you wish to debug or run a single condition manually without command-line arguments:

    1. Open the specific script (e.g., `train.py`).

    2. Locate the `main` function.

    3. Uncomment the hardcoded parameter block (marked as check/check point).

    4. Run the script directly as before.

##  8. Anlaysis: `5_FittingCurveAnalysis`

### 8.1 Function Specification

Cumulative Gaussian 进行非线性最小二乘拟合；使用 `erf` 为误差函数，反映被试在不同刺激强度下的反应概率。

$$y = g + (1 - g - l) \cdot \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x - u}{\sqrt{2v^2}} \right) \right]$$

* $g$: Guess Rate，对于 2AFC 实验，通常固定为0.5；对于其他 n-AFC，通常为 $1/n$
* $l$: Lapse Rate失误率，反映高强度刺激下的非感知性错误
* $u$: PSE or Threshold，主观相等点或感知阈值
* $v$: Sigma, 噪声参数，决定曲线斜率


### 8.2 Variable

#### Input

| 符号 | 定义 |
| :--- | :--- |
| `xAxis` | 刺激强度 (Stimulus Intensity) |
| `yData` | 反应正确率 (Proportion Correct) |

#### Parameter
| 符号 | 定义 |
| :--- | :--- |
| `u` | PSE / 阈值 (累积函数中心点) |
| `v` | 标准差 (控制曲线平滑度) |
| `g` | 猜测率 (下限值) |
| `l` | 失误率 (上限偏移) |

#### Output
| 符号 | 定义 |
| :--- | :--- |
| `slope` | 中点斜率 (敏感度) |
| `curve` | 高分辨率拟合曲线 |
| `ffit` | 拟合对象 (含 $R^2$, 残差) |



### 8.3 How to Compile

**Run** - `PsychometricFittingCurve.m`

**Automation Steps performed by the script:**

* Auto-Fetch Data: Checks ../ConvResults and automatically copies relevant CSVs to ../Data/Conv.

* Adaptive Loading: Automatically detects if data has 21, 11, or 9 rows and maps them to the correct stimulus intensity points.

* Fitting & Stats: Calculates PSE/Slope, runs T-tests against human data, calculates Cohen's d.

* Plotting: Generates smoothed psychometric curves for all models and human data.

**Output** - Originally Saved in Folder `Results`

### 8.4 Notes

#### About Constraints

无论是行为学实验还是model的任务，都要求被试/model在“快乐”与“悲伤”中做出分类判断，因此本实验属于 **2AFC Discrimination** Task （2AFC辨别任务）；此外，y轴被定义为Proportion of Happy Response。在极端悲伤 (Intensity=0) 的情况下，判定为“快乐”的概率应趋近于 0 (False Alarm Rate $\approx$ 0)，而非 0.5。

因此对于constraints, 本研究使用：

* *GuessRate* （$g$）[0, 0.05]：极端非快乐点误按“快乐”的概率，限制虚报率在 5% 以内。

* *LapseRate* （$l$） [0, 0.05]：极端快乐点因走神导致的漏报概率。上限设为心理物理学标准 0.05。

* *PSE* （$u$） [0, 1]：50%响应时的强度（分类边界）。预期值（humandata）约 0.51。0.5 为真正的中性判定点。

* *Slope* [0.001, 100]：知觉切换的敏锐度，允许极高的分类精度。预期值（humandata）约 4.19。

In Code ~Line 24~27：

``` Matlab
% Para Limitation
UL = [0.05, 0.05, 1, 100]; 
SP = [0.01, 0.02, 0.5, 5.0];  
LM = [0, 0, 0, 0.1]; 
```

注：其他任务
* Yes/No 任务：$ g \approx 0$，反映信号检测阈限（False Alarm）；
* 2AFC 检测：$ g = 0.5 $，曲线从半山腰开始，反映猜测正确率。

#### About Line / FittingCurve

共有三组地方需要改动（SE-AlexNet, AlexNet, VGG），每组地方有两处需要修改；以VGG16这一组为例↓：

``` Matlab
% Convert "curveV" into "~" if you want to draw LINE CHART for the Models 
            [ffit, ~, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM); %Comment this Line if you are drawing CURVE
            [ffit, curveV, slope] = FitPsycheCurveWH(paraX', ySampled, UL, SP, LM); %Comment this Line if you are drawing LINE chart

% Change parameters for plotting if you want to draw LINE CHART for the Models
            plot(paraX, ySampled, '-o', 'Color', colors.vgg16, 'MarkerSize', 5, ...
                'MarkerFaceColor', colors.vgg16, 'LineWidth', 2.5, 'DisplayName', 'VGG 16'); %Comment this Line if you are drawing CURVE
            plot(curveV(:,1), curveV(:,2), '-', 'Color', colors.vgg16, 'LineWidth', 2.5, ...
                'HandleVisibility', 'on', 'DisplayName', 'VGG 16'); %Comment this Line if you are drawing LINE chart
            plot(paraX, ySampled, 'o', 'MarkerEdgeColor', colors.vgg16, 'MarkerFaceColor',...
                 colors.vgg16, 'MarkerSize', 5, 'HandleVisibility', 'off'); %Comment this Line if you are drawing LINE chart
```
#### Statistical Analysis

Added **One-Sample T-Test** to compare Model PSE vs. Human Population PSEs.

* Human Baseline: Fitted individual PSEs for 50 participants to derive Mean & SD.

* Metrics Exported (in Final_Analysis_Results.xlsx):

    * t_stat: T-value
    * p_value: Significance level
    * Cohens_d: Effect size ($d = |PSE_{model} - Mean_{human}| / SD_{human}$)

#### Naming & Data Management

Using locConfigs structure to manage mixed naming conventions:

1. Old Format (Hyphen): `Location-1-FaceBased-E-2.csv`

    Mapped to Display Name: SE-FC-L1, SE-FC-L2, SE-FC-L3

2. New Format (Underscore): `SeC1_FaceBased_squeeze2_Full.csv`

    Mapped to Display Name: SE-Conv-L1, SE-Conv-L2, SE-Conv-L3, SE-Conv-L4

This ensures clean plot titles and Excel reports despite different underlying file naming rules.
