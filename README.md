# SE-AlexNet

SE-AlexNet 模型代码、训练指标与 21 点预测存档。

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `0_InputImages/` | 每个刺激条件的 21 张面孔图像 |
| `1_Codes/` | AlexNet、VGG16、SE-AlexNet（三种 FC 插入位置）模型与训练代码 |
| `2_Results/1_Predict21Datapoints/` | 136 个预测文件：120 个 SE-AlexNet 条件 + 16 个 AlexNet/VGG16 基线 |
| `3_Ablation(Codes)/` `4_Ablation(Results)/` | 卷积层插入的消融实验 |

## 校验预测存档

```bash
pip install -r requirements.txt
python3 scripts/validate_prediction_archive.py
```

检查 120 个 SE-AlexNet + 16 个基线文件、每个文件恰好 21 个 [0,1] 概率，并生成 `2_Results/Prediction_Archive_Manifest.csv`（含每个文件的行数、取值范围与 SHA-256 哈希）。

## 复现心理物理分析

预测存档同时收录于 [PsychometricFittingCurve](https://github.com/JyBmegan/PsychometricFittingCurve)，在那里运行 `reproduce_psychometric_results.py` 即得到论文的 PSE / ANOVA 表。

## 训练模型

训练权重发布在 [Hugging Face](https://huggingface.co/JiayuMBao/SE-AlexNet)。设置 `1_Codes/train.py` 中的数据集与预训练权重路径后运行：

```bash
python3 1_Codes/train.py
```

## 预测文件命名

`Location-{1,2,3}-{FaceBased,ObjectBased}-{Full,E,N,M}-{2,4,8,16,32}.csv`

- 插入位置：`Location-1` 前序（FC6 之前）、`Location-2` 中序（FC6 与 FC7 之间）、`Location-3` 后序（FC7 与 FC8 之间）。
- 预训练：`FaceBased` 基于面孔、`ObjectBased` 基于物体。
- 掩码：`Full` 完整、`E` 遮眼、`N` 遮鼻、`M` 遮嘴。
- 压缩比：2、4、8、16、32。

基线文件命名为 `{AlexNet,VGG-16}-{FaceBased,ObjectBased}-{Full,E,N,M}-raw.csv`。

## SE 模块

SE 模块对通道特征做「压缩-激励」重标定：先全局平均池化得到通道向量 $z$，再经两个全连接层（中间降到 $C/r$ 维）得到通道权重，与原特征逐通道相乘：

$$
s=\sigma\big(W_2\,\operatorname{ReLU}(W_1 z)\big),\qquad \hat{x}=x\odot s
$$

其中 $r$ 为压缩比，$C/r$ 是压缩后的维度（例如 FC 层 4096 维、$r=16$ 时压缩到 256 维）。
