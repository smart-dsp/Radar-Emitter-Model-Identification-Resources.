
# 雷达调制类型识别资源汇总

<p align="center">
  <b>简体中文</b> |
  <a href="./README.en.md">English</a>
</p>

本仓库持续整理 **Radar Modulation Type Recognition / Radar Waveform Recognition / Automatic Intrapulse Modulation Classification（雷达调制类型识别 / 雷达波形识别 / 雷达脉内调制识别）** 相关的开源资源，重点关注任务定义、公开数据集、开源实现、代表性方法、评价指标、可复现实验设置以及相关任务边界。

Radar modulation type recognition 的目标是：给定非合作接收机截获到的一段雷达信号，通常为复基带 **I/Q 序列**、时频图、频谱图或其他特征表示，识别其采用的雷达波形或脉内调制类型，例如 **LFM、NLFM、SFM、Costas、Barker、Frank、P1/P2/P3/P4、BPSK、QPSK、FSK、OFDM、FMCW** 等。

目前该方向的开源资源比较分散，不同项目对 `radar waveform recognition`、`LPI radar recognition`、`radar signal recognition`、`automatic intrapulse modulation classification`、`radar modulation classification` 等概念的使用并不完全一致。因此，本仓库尝试对相关资源进行人工整理，并标注其任务类型、数据可用性、监督方式、是否严格符合“雷达调制类型识别”定义，以及是否容易复现。

---

## Highlights / 快速结论

* **标准 benchmark 首选：** AIMC-Spec、DeepRadar2022、RadChar / RadCharSSL。
* **经典复现起点：** LPI-Radar-Waveform-Recognition，采用 CWD 时频分析 + CNN / LPI-Net。
* **数据生成起点：** Radar-Intra-Pulse-Modulation-Signal-Simulation 与 Radar-Intra-Pulse-Modulation-Simulation。
* **低信噪比鲁棒识别：** DNCNet、SEMTN、SFUnet-DCNN 等方向值得关注。
* **少样本 / 自监督方向：** RadCharSSL、CTNet-SSL、few-shot radar signal recognition 相关代码。
* **扩展任务：** 多分量雷达信号识别、参数估计、信号解析、雷达 + 通信联合识别等与调制识别密切相关，但不应与标准单信号调制分类混为一类。
* **需要谨慎区分：** 通信 AMC、雷达目标识别、雷达检测、脉冲分选、雷达辐射源个体识别、RF fingerprinting 与雷达调制类型识别不是同一个任务。

> 注：本仓库中的 ⭐ 推荐程度是整理者基于任务匹配度、开源程度、复现价值和数据说明完整度给出的主观推荐，不代表 GitHub stars。

---

## 目录 / Table of Contents

* [1. 任务概述 / Overview](#1-任务概述--overview)

  * [1.1 什么是雷达调制类型识别？](#11-什么是雷达调制类型识别)
  * [1.2 常见输入表示](#12-常见输入表示)
  * [1.3 常见调制类别](#13-常见调制类别)
  * [1.4 常见评价指标](#14-常见评价指标)
* [2. 方法分类与任务边界 / Method Taxonomy and Task Fit](#2-方法分类与任务边界--method-taxonomy-and-task-fit)

  * [2.1 传统特征 + 分类器方法](#21-传统特征--分类器方法)
  * [2.2 时频图 + 深度学习方法](#22-时频图--深度学习方法)
  * [2.3 一维 I/Q 序列端到端方法](#23-一维-iq-序列端到端方法)
  * [2.4 去噪 / 增强 + 识别方法](#24-去噪--增强--识别方法)
  * [2.5 少样本、自监督与域适应方法](#25-少样本自监督与域适应方法)
  * [2.6 度量学习、开放集与未知类识别](#26-度量学习开放集与未知类识别)
  * [2.7 哪些方法严格符合调制识别定义？](#27-哪些方法严格符合调制识别定义)
* [3. 数据集资源 / Datasets](#3-数据集资源--datasets)

  * [3.1 数据集总览](#31-数据集总览)
  * [3.2 重点数据集说明](#32-重点数据集说明)
* [4. 开源方法与实现 / Methods and Implementations](#4-开源方法与实现--methods-and-implementations)

  * [4.1 方法与代码总览](#41-方法与代码总览)
  * [4.2 重点方法说明](#42-重点方法说明)
* [5. 推荐实验设置 / Recommended Experimental Setup](#5-推荐实验设置--recommended-experimental-setup)

  * [5.1 主 benchmark](#51-主-benchmark)
  * [5.2 基线方法](#52-基线方法)
  * [5.3 建议流程](#53-建议流程)
* [6. 相关任务 / Related Tasks](#6-相关任务--related-tasks)
* [7. 推荐阅读与入门资源 / Recommended Reading and Starting Points](#7-推荐阅读与入门资源--recommended-reading-and-starting-points)
* [8. 说明 / Notes](#8-说明--notes)
* [9. 引用与贡献 / Citation and Contribution](#9-引用与贡献--citation-and-contribution)

---

## 1. 任务概述 / Overview

### 1.1 什么是雷达调制类型识别？

雷达调制类型识别，也常被称为 **雷达波形识别**、**LPI 雷达波形识别**、**雷达信号调制识别** 或 **自动脉内调制分类（Automatic Intrapulse Modulation Classification, AIMC）**。

给定一段截获的雷达信号：

```text
x = {x1, x2, ..., xN}, xi ∈ C
```

其中 `x` 通常为复基带 I/Q 采样序列，任务目标是预测该信号对应的调制或波形类别：

```text
y ∈ {LFM, NLFM, SFM, Costas, Barker, Frank, P1, P2, P3, P4, BPSK, QPSK, FSK, FMCW, ...}
```

从最常见的研究设定看，雷达调制类型识别通常是一个 **监督式多分类问题（supervised multi-class classification problem）**。训练阶段使用带有调制标签的数据，测试阶段对未知信号输出调制类别。

不过，在更复杂的电子侦察和电子支援场景中，该任务也可能扩展为：

* 低信噪比下的鲁棒调制识别；
* 少样本调制识别；
* 未知调制开放集识别；
* 多分量 / 重叠雷达信号识别；
* 调制类型识别 + 参数估计；
* 雷达信号语义解析；
* 雷达与通信信号联合识别。

需要注意，雷达调制类型识别 **不是** 雷达目标识别，也不是雷达脉冲分选。前者识别的是波形或脉内调制类别，后者通常关注目标类别、辐射源簇、脉冲归属或设备身份。

---

### 1.2 常见输入表示

| Input 表示                  | Description 说明        | Usage 用途                            |
| ------------------------- | --------------------- | ----------------------------------- |
| I/Q sequence              | 复基带同相 / 正交采样序列        | 最直接的输入形式，适合 1D-CNN、LSTM、Transformer |
| Amplitude / phase         | 幅度、相位或瞬时相位            | 可用于传统特征、相位编码识别                      |
| Instantaneous frequency   | 瞬时频率                  | 对 LFM、NLFM、SFM、FSK 等频率调制有帮助         |
| Spectrum                  | 频谱或功率谱                | 适合粗粒度波形识别                           |
| Spectrogram               | STFT 频谱图              | 图像分类方法常用输入                          |
| Time-frequency image, TFI | WVD、CWD、CWT、FSST 等时频图 | LPI 雷达识别中非常常见                       |
| Ambiguity function        | 模糊函数                  | 可反映雷达波形的时延-多普勒结构                    |
| Multi-modal features      | I/Q + 时频图 + 统计特征      | 适合多模态融合、少样本学习                       |

---

### 1.3 常见调制类别

不同数据集和论文使用的类别集合不完全一致，常见类别包括：

| Family 类别族                  | Examples 示例                                         |
| --------------------------- | --------------------------------------------------- |
| Unmodulated / CW            | CW, pulse, unmodulated                              |
| Frequency modulation        | LFM, NLFM, SFM, FMCW, triangular FM, exponential FM |
| Frequency shift coding      | BFSK, 2FSK, 4FSK, Costas                            |
| Phase coding                | BPSK, QPSK, Barker, Frank, P1, P2, P3, P4           |
| Polyphase coding            | Frank, P1, P2, P3, P4, Zadoff-Chu-like codes        |
| OFDM-like waveform          | OFDM, multi-carrier radar waveform                  |
| Composite / multi-component | dual-LFM, multi-LFM, overlapping radar signals      |
| Radar-communication mixed   | radar waveform + communication modulation           |

---

### 1.4 常见评价指标

| Metric 指标                  | Description 说明         |
| -------------------------- | ---------------------- |
| Accuracy                   | 分类正确率，最常用指标            |
| Macro-F1                   | 类别不均衡时更有参考价值           |
| Precision / Recall         | 分类别分析误报和漏报             |
| Confusion matrix           | 分析哪些调制类型容易混淆           |
| Accuracy vs. SNR           | 低信噪比鲁棒性分析核心指标          |
| Open-set AUROC             | 开放集 / 未知类识别常用指标        |
| Few-shot accuracy          | N-way K-shot 设置下的识别准确率 |
| Cross-dataset accuracy     | 跨数据集泛化能力指标             |
| Parameter estimation error | 当任务包含参数估计时使用           |

---

## 2. 方法分类与任务边界 / Method Taxonomy and Task Fit

雷达调制类型识别方法大致可以分为 **传统特征 + 分类器、时频图 + 深度学习、一维 I/Q 端到端模型、去噪增强 + 识别、少样本 / 自监督 / 域适应、度量学习 / 开放集识别、多任务解析** 等几类。

---

### 2.1 传统特征 + 分类器方法

这类方法通常先从 I/Q 信号中提取人工设计特征，再使用传统机器学习分类器完成识别。

| Method 方法                        | Main idea 核心思想              | Advantages 优点 | Limitations 局限 |
| -------------------------------- | --------------------------- | ------------- | -------------- |
| Instantaneous feature + SVM      | 提取瞬时频率、瞬时相位、包络等特征后分类        | 可解释性强         | 对噪声和参数变化敏感     |
| Higher-order cumulants           | 利用高阶统计特征区分调制类型              | 对部分调制有效       | 对复杂雷达波形覆盖不足    |
| Time-frequency feature + KNN/SVM | 从 STFT/WVD/CWD 等图中提取纹理或形状特征 | 适合 LPI 波形     | 特征设计依赖经验       |
| Sparse representation            | 用稀疏字典表示不同波形                 | 理论清晰          | 计算成本较高         |
| Template matching                | 与已知调制模板或参数模板匹配              | 适合已知波形        | 泛化能力有限         |

传统方法适合作为可解释 baseline，但在低 SNR、参数变化、未知信道、多径和跨数据集场景下通常不如深度学习方法稳健。

---

### 2.2 时频图 + 深度学习方法

这类方法是目前雷达调制识别中最常见的深度学习路线。典型流程是：

```text
I/Q signal
    ↓
Time-frequency transform
    ↓
TFI / spectrogram image
    ↓
CNN / ResNet / DenseNet / ViT / Transformer
    ↓
Modulation class
```

| Method Family 方法族                | Main idea 核心思想                      | 备注             |
| -------------------------------- | ----------------------------------- | -------------- |
| STFT + CNN                       | 将信号转为 spectrogram，再进行图像分类           | 简单、易复现         |
| CWD + CNN                        | 使用 Choi-Williams Distribution 获得时频图 | LPI-Net 代表路线   |
| WVD / PWVD + CNN                 | 使用 Wigner-Ville 系列时频分布              | 分辨率高，但交叉项问题明显  |
| CWT + CNN                        | 使用连续小波变换表示多尺度结构                     | 对非平稳信号友好       |
| FSST / synchrosqueezing + CNN    | 更精细的时频重排                            | 预处理复杂          |
| ResNet / DenseNet / EfficientNet | 使用成熟图像分类网络                          | 容易建立强 baseline |
| ViT / Swin / CNN-Transformer     | 使用注意力建模全局结构                         | 数据量要求较高        |

---

### 2.3 一维 I/Q 序列端到端方法

这类方法直接使用 I/Q 序列作为输入，避免时频图预处理。

| Method 方法                     | Main idea 核心思想        | 备注                  |
| ----------------------------- | --------------------- | ------------------- |
| 1D-CNN                        | 对 I/Q 双通道序列进行卷积特征提取   | 简单高效                |
| CNN-LSTM / CNN-GRU            | CNN 提取局部特征，RNN 建模时间依赖 | 适合时序结构明显的信号         |
| TCN                           | 使用时间卷积网络建模长距离依赖       | 比 RNN 更易并行          |
| Transformer encoder           | 用自注意力建模全局序列关系         | 对数据量和正则化要求较高        |
| Complex-valued neural network | 直接建模复数信号              | 理论上更贴近 I/Q 数据，但实现复杂 |

一维方法的优势是输入更接近原始信号，缺点是可解释性弱，且不同采样率、脉宽、带宽、载频偏移等因素会显著影响泛化。

---

### 2.4 去噪 / 增强 + 识别方法

低 SNR 是雷达调制识别中的核心难点。常见做法包括：

| Method 方法                      | Main idea 核心思想   | 代表方向                    |
| ------------------------------ | ---------------- | ----------------------- |
| Denoising + classifier         | 先去噪，再分类          | DNCNet                  |
| Time-frequency enhancement     | 对时频图进行去噪、增强或重构   | SFUnet-DCNN、TFGM 等      |
| Residual / attention network   | 用残差结构、通道注意力增强鲁棒性 | ResNet、SE、CBAM、MAPNet   |
| Multi-scale feature extraction | 多尺度捕获调制纹理        | MAPNet、pyramid networks |
| Data augmentation              | 加噪声、频偏、时间偏移、幅度扰动 | 提升泛化能力                  |

---

### 2.5 少样本、自监督与域适应方法

真实雷达数据难以获取，标注成本高，因此少样本和自监督学习越来越重要。

| Method 方法                                    | Main idea 核心思想          | Supervision level 监督程度 |
| -------------------------------------------- | ----------------------- | ---------------------- |
| Few-shot learning                            | N-way K-shot 设置下识别新调制类别 | 少量标注                   |
| MAML / meta-learning                         | 学习可快速适应新任务的初始化          | 少样本监督                  |
| Contrastive learning                         | 构造正负样本，学习判别性表示          | 自监督 / 弱监督 / 监督均可能      |
| Masked signal modeling                       | 遮蔽部分 I/Q 或时频图后重构        | 自监督                    |
| Domain adaptation                            | 从通信或其他 RF 数据迁移到雷达域      | 无监督 / 半监督 / 少监督        |
| Self-supervised pretraining + linear probing | 先预训练表示，再少量标注微调          | 自监督 + 少量监督             |

---

### 2.6 度量学习、开放集与未知类识别

在电子侦察中，测试阶段可能出现训练集中不存在的调制类型。因此开放集识别和度量学习也很重要。

| Method 方法                       | Main idea 核心思想         | 备注             |
| ------------------------------- | ---------------------- | -------------- |
| Triplet loss                    | 同类拉近，异类推远              | 训练阶段需要标签       |
| Contrastive loss                | 学习相似 / 不相似信号对          | 可监督或自监督        |
| Prototypical network            | 每类学习原型中心               | few-shot 常用    |
| Metric learning + clustering    | 学到 embedding 后聚类或最近邻分类 | 适合新类别扩展        |
| OpenMax / thresholding          | 通过置信度或距离阈值拒识未知类        | 简单开放集 baseline |
| Energy-based open-set detection | 使用能量分数检测未知样本           | 深度开放集方向        |

---

### 2.7 哪些方法严格符合调制识别定义？

> **严格定义：** 给定一段单雷达信号或单脉冲 I/Q / 时频图输入，输出其雷达调制或波形类别。

| Method Family 方法族                        | Matches the task? 是否符合调制识别任务 | Strict classification? 是否标准分类 | Recommendation 推荐程度 | Comment 说明                 |
| ---------------------------------------- | ---------------------------- | ----------------------------- | ------------------- | -------------------------- |
| STFT/CWD/WVD/CWT + CNN                   | 是                            | 是                             | ⭐⭐⭐⭐⭐               | 当前最常见、最容易复现                |
| LPI-Net / CWD-TFA + CNN                  | 是                            | 是                             | ⭐⭐⭐⭐⭐               | 经典 LPI 雷达波形识别路线            |
| AIMC-Spec image classification           | 是                            | 是                             | ⭐⭐⭐⭐⭐               | 推荐作为 spectrogram benchmark |
| 1D-CNN / LSTM / Transformer on I/Q       | 是                            | 是                             | ⭐⭐⭐⭐                | 适合端到端基线                    |
| Denoising + classification               | 是                            | 是                             | ⭐⭐⭐⭐                | 适合低 SNR 场景                 |
| Few-shot / SSL radar recognition         | 是                            | 是 / 少样本                       | ⭐⭐⭐⭐                | 适合标注不足场景                   |
| Metric learning / open-set recognition   | 是                            | 扩展分类                          | ⭐⭐⭐⭐                | 适合未知类和鲁棒识别                 |
| Multi-component radar signal recognition | 部分符合                         | 多标签 / 多实例                     | ⭐⭐⭐                 | 比标准单信号分类更难                 |
| Parameter estimation / Sig2text          | 部分符合                         | 多任务 / 解析                      | ⭐⭐⭐                 | 调制识别的扩展任务                  |
| RadarComm joint classification           | 部分符合                         | 混合雷达通信分类                      | ⭐⭐⭐                 | 需区分雷达标签和通信标签               |
| Communication AMC                        | 相关但不等同                       | 通信调制分类                        | ⭐⭐                  | 不能直接等同于雷达调制识别              |
| Radar pulse deinterleaving               | 不同任务                         | 聚类 / 分选                       | ⭐⭐                  | 识别脉冲归属，不识别调制类型             |
| Radar target recognition / SAR ATR       | 不同任务                         | 目标分类                          | ⭐                   | 识别目标，不识别波形调制               |

---

## 3. 数据集资源 / Datasets

本节整理雷达调制类型识别相关的数据集、仿真数据、数据生成代码和相关任务数据。推荐程度主要根据 **任务匹配度、是否公开、标签是否清晰、是否适合作为 benchmark、数据说明是否完整、复现实验是否便利** 等因素综合判断。

---

### 3.1 数据集总览

| Dataset 数据集                             | Task Fit 任务匹配度     | Data Type 数据类型                       | Labels 标签            | Open Source? 是否开源 | Recommendation 推荐度 | Links 链接                                                                                                                                                                                | Notes 备注                   |
| --------------------------------------- | ------------------ | ------------------------------------ | -------------------- | ----------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| AIMC-Spec / AIMC-Spec-v2                | 标准脉内调制分类           | Spectrogram / image                  | 有调制类别标签              | 是                 | ⭐⭐⭐⭐⭐              | [Paper](https://arxiv.org/html/2601.08265v2) / [GitHub](https://github.com/seb-cocks/AIMC-image-classification) / [Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2) | 论文版与代码版类别数可能有差异，使用时需注明版本   |
| DeepRadar2022                           | 雷达调制分类             | I/Q sequence                         | 23 类雷达调制             | 是                 | ⭐⭐⭐⭐⭐              | [Kaggle](https://www.kaggle.com/datasets/khilian/deepradar)                                                                                                                             | 适合作为 1D I/Q 序列分类 benchmark |
| RadChar                                 | 雷达信号表征 / 多任务识别     | I/Q sequence, HDF5                   | 调制类别 + 参数标签          | 是                 | ⭐⭐⭐⭐               | [GitHub](https://github.com/abcxyzi/RadChar)                                                                                                                                            | 不只是分类，还支持参数回归              |
| RadCharSSL                              | 少样本 / 自监督雷达识别      | I/Q datasets                         | few-shot / SSL 设置    | 是                 | ⭐⭐⭐⭐               | [GitHub](https://github.com/abcxyzi/RadCharSSL) / [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)                                                                | 适合自监督预训练和少样本实验             |
| RadarCommDataset                        | 雷达 + 通信联合识别        | I/Q waveform                         | 雷达 / 通信 / 调制多标签      | 是                 | ⭐⭐⭐                | [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset)                                                                                                               | 混合数据集，不是纯雷达调制分类            |
| LPI-Net generated data                  | LPI 雷达波形识别         | Synthetic radar waveform + CWD TFI   | LPI waveform labels  | 可生成               | ⭐⭐⭐⭐               | [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)                                                                                                               | 适合复现 CWD + CNN pipeline    |
| Liuyh0308 simulation data               | 雷达脉内调制识别           | `.mat` I/Q + WVD/CWD/STFT/CWT images | 有调制类别                | 可生成               | ⭐⭐⭐⭐               | [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)                                                                                                   | 适合 few-shot 和多模态融合实验       |
| DNCNet SIGNAL-8                         | 低 SNR 雷达信号识别       | 1D radar signals                     | 8 类雷达信号              | 仓库内提供 / 可复现       | ⭐⭐⭐⭐               | [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)                                                                                                                                | 适合去噪 + 识别任务                |
| SEMTN data                              | 多信道 / 多 SNR 雷达调制识别 | `.mat` data                          | 调制类别 + SNR / channel | 仓库内提供             | ⭐⭐⭐                | [GitHub](https://github.com/Guqih/SEMTN)                                                                                                                                                | README 简略，需自行检查字段          |
| MAPNet data                             | 鲁棒雷达信号识别           | `.npy` / model weights               | 调制类别                 | 部分公开              | ⭐⭐⭐                | [GitHub](https://github.com/bryantky/MAPNet)                                                                                                                                            | 复现前需核查数据完整性                |
| Radar-Intra-Pulse-Modulation-Simulation | 数据生成               | Python synthetic signals             | 可自行生成标签              | 代码可生成             | ⭐⭐⭐                | [GitHub](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)                                                                                                       | 适合教学和补充仿真                  |
| RadioML / DeepSig                       | 通信 AMC             | Communication I/Q                    | 通信调制标签               | 是                 | ⭐⭐                 | [RadioML example](https://www.kaggle.com/datasets/pinxau1000/radioml2018)                                                                                                               | 不是雷达调制识别，只能作为迁移学习或对照       |
| RadDet                                  | 雷达频谱检测             | Wideband spectrum                    | 检测标签                 | 是                 | ⭐⭐                 | [GitHub](https://github.com/abcxyzi/RadDet)                                                                                                                                             | 相关任务，不是调制分类                |
| RadSeg                                  | 雷达活动分割             | Long I/Q sequence                    | sample-wise labels   | 是                 | ⭐⭐                 | [GitHub](https://github.com/abcxyzi/radseg)                                                                                                                                             | 相关任务，不是标准调制识别              |

---

### 3.2 重点数据集说明

#### 3.2.1 AIMC-Spec / AIMC-Spec-v2

**推荐程度：** ⭐⭐⭐⭐⭐
**适合用途：** 标准脉内调制分类 benchmark、spectrogram 图像分类、统一模型评估
**数据类型：** 雷达脉内调制信号的频谱图 / 时频图图像
**标签情况：** 有调制类别标签
**注意事项：** 论文版和 GitHub / Kaggle 版本可能存在类别数量差异，实验时需注明具体版本。

AIMC-Spec 是目前非常适合作为自动脉内调制分类的标准化 benchmark。它将雷达调制识别任务组织为基于 spectrogram 的图像分类问题，适合复现 CNN、ResNet、轻量网络、去噪网络和 Transformer 等模型。

**链接：** [Paper](https://arxiv.org/html/2601.08265v2) / [GitHub](https://github.com/seb-cocks/AIMC-image-classification) / [Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)

---

#### 3.2.2 DeepRadar2022

**推荐程度：** ⭐⭐⭐⭐⭐
**适合用途：** 一维 I/Q 雷达调制分类、LSTM / CNN / Transformer 基线
**数据类型：** 1024 × 2 I/Q 序列
**标签情况：** 23 类雷达调制标签
**注意事项：** 适合直接做 1D 模型，不需要先转成图像。

DeepRadar2022 是较常用的雷达调制分类数据集之一，覆盖多种雷达调制类型，适合做 I/Q 序列端到端识别实验。

**链接：** [Kaggle](https://www.kaggle.com/datasets/khilian/deepradar)

---

#### 3.2.3 RadChar

**推荐程度：** ⭐⭐⭐⭐
**适合用途：** 雷达信号表征、多任务学习、分类 + 参数估计
**数据类型：** synthetic radar I/Q data
**标签情况：** 同时提供分类和回归标签
**注意事项：** 它不是单纯的调制分类数据集，更适合 radar signal characterisation。

RadChar 的价值在于不仅提供调制类别标签，还提供参数估计相关标签，因此适合研究“调制识别 + 参数估计”的多任务模型。

**链接：** [GitHub](https://github.com/abcxyzi/RadChar)

---

#### 3.2.4 RadCharSSL

**推荐程度：** ⭐⭐⭐⭐
**适合用途：** 少样本雷达信号识别、自监督预训练、RF domain adaptation
**数据类型：** 雷达 I/Q 数据与少样本划分
**标签情况：** 支持 few-shot / SSL 实验设置
**注意事项：** 适合研究标注稀缺场景，不一定适合作为普通 supervised classification 的唯一 benchmark。

RadCharSSL 面向 few-shot radar signal recognition 与 self-supervised learning，适合研究 masked signal modeling、domain adaptation、linear probing 和少样本微调。

**链接：** [GitHub](https://github.com/abcxyzi/RadCharSSL) / [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)

---

#### 3.2.5 RadarCommDataset

**推荐程度：** ⭐⭐⭐
**适合用途：** 雷达 + 通信联合识别、ISAC / 频谱共存、跨域学习
**数据类型：** 雷达和通信 I/Q 波形
**标签情况：** 多任务标签，包括信号类型和调制类型
**注意事项：** 不是纯雷达调制识别数据集，需要区分 radar waveform 和 communication modulation。

RadarCommDataset 同时包含雷达和通信波形，适合研究联合分类、域适应和跨任务迁移，但不应直接作为纯雷达调制识别 benchmark 使用。

**链接：** [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset)

---

#### 3.2.6 LPI-Net generated data

**推荐程度：** ⭐⭐⭐⭐
**适合用途：** LPI 雷达波形识别、CWD 时频图、CNN 分类
**数据类型：** synthetic radar waveform + CWD time-frequency image
**标签情况：** 有 LPI waveform labels
**注意事项：** 更适合复现经典 pipeline，而不是直接当作标准大规模 benchmark。

该项目的价值在于流程完整：雷达信号生成、CWD 时频分析、图像转换、CNN 训练与测试。

**链接：** [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)

---

#### 3.2.7 Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation

**推荐程度：** ⭐⭐⭐⭐
**适合用途：** 脉内调制信号仿真、few-shot、多模态融合、时频图生成
**数据类型：** `.mat` 序列 + WVD / CWD / STFT / CWT images
**标签情况：** 可生成调制类别标签
**注意事项：** 数据主要通过代码生成，复现前需确认参数设置。

该项目适合构建自己的雷达脉内调制识别数据集，也适合比较一维 I/Q、时频图、多模态融合方法。

**链接：** [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)

---

## 4. 开源方法与实现 / Methods and Implementations

本节整理雷达调制类型识别相关的开源方法、代码实现和可复现实验框架。推荐程度主要根据 **任务匹配度、是否开源、是否容易复现、是否适合作为 baseline、方法代表性、数据说明完整度** 等因素综合判断。

---

### 4.1 方法与代码总览

| Project / Method 项目或方法                                   | Method Type 方法类型                              | Input 输入                      | Supervision 监督方式 | Strictly RMR? 是否严格属于雷达调制识别 | Recommendation 推荐度 | Links 链接                                                                                                           | Notes 备注                       |
| -------------------------------------------------------- | --------------------------------------------- | ----------------------------- | ---------------- | -------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| LPI-Radar-Waveform-Recognition / LPI-Net                 | CWD + CNN                                     | Time-frequency image          | 监督               | 是                          | ⭐⭐⭐⭐⭐              | [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)                                          | 经典入门复现项目                       |
| AIMC-image-classification                                | Spectrogram + deep models                     | Spectrogram image             | 监督               | 是                          | ⭐⭐⭐⭐⭐              | [GitHub](https://github.com/seb-cocks/AIMC-image-classification)                                                   | AIMC-Spec 官方/配套框架              |
| Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation | Signal simulation + TFI + few-shot            | I/Q + time-frequency images   | 监督 / few-shot    | 是                          | ⭐⭐⭐⭐               | [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)                              | 数据生成和多模态实验价值高                  |
| DNCNet-pytorch                                           | Denoising + classification                    | 1D radar signal               | 监督               | 是                          | ⭐⭐⭐⭐               | [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)                                                           | 低 SNR 鲁棒识别                     |
| MAPNet                                                   | Multi-scale atrous pyramid + metric learning  | Radar signal representation   | 监督度量学习           | 是                          | ⭐⭐⭐⭐               | [GitHub](https://github.com/bryantky/MAPNet)                                                                       | 适合鲁棒识别和开放集方向                   |
| CTNet-SSL                                                | CNN-Transformer + contrastive SSL             | Radar signal                  | 自监督 / 对比学习       | 是                          | ⭐⭐⭐                | [GitHub](https://github.com/wanan0414/CTNet-SSL)                                                                   | 文档相对较少，需核查复现细节                 |
| SEMTN                                                    | Self-enhanced Multidimensional Taylor Network | `.mat` radar data             | 监督               | 是                          | ⭐⭐⭐                | [GitHub](https://github.com/Guqih/SEMTN)                                                                           | 多信道、多 SNR 数据，但 README 简略       |
| rsrc-for-pub                                             | Multiple radar recognition resources          | TFI / signal / datasets       | 多种               | 部分是                        | ⭐⭐⭐⭐               | [GitHub](https://github.com/stu-cjlu-sp/rsrc-for-pub)                                                              | 包含 SFUnet-DCNN、SQDandFE 等多个子项目 |
| RadarCNN                                                 | CNN on DeepRadar2022                          | I/Q or image                  | 监督               | 是                          | ⭐⭐                 | [GitHub](https://github.com/blackmirag3/RadarCNN)                                                                  | hobby / demo 性质，适合快速参考         |
| Radar-Intra-Pulse-Modulation-Simulation                  | Signal simulation                             | I/Q                           | 数据生成             | 部分符合                       | ⭐⭐⭐                | [GitHub](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)                                  | 只生成信号，不是完整识别框架                 |
| Sig2text                                                 | Vision-language / signal-to-text              | Time-frequency representation | 监督 / 解析          | 扩展任务                       | ⭐⭐⭐                | [Paper](https://arxiv.org/abs/2503.15213)                                                                          | 调制识别 + 参数估计 + 符号解析             |
| MathWorks LPI radar waveform classification              | Time-frequency CNN example                    | Synthetic waveform + TFI      | 监督               | 是                          | ⭐⭐⭐                | [Example](https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html) | 教学价值高，但不是普通 GitHub 开源项目        |
| RadioML / communication AMC repos                        | Communication AMC                             | Communication I/Q             | 监督               | 否                          | ⭐⭐                 | [Example](https://github.com/topics/modulation-classification)                                                     | 只能作为迁移学习或对照，不能直接当雷达调制识别        |

---

### 4.2 重点方法说明

#### 4.2.1 LPI-Radar-Waveform-Recognition / LPI-Net

**推荐程度：** ⭐⭐⭐⭐⭐
**方法类型：** CWD time-frequency analysis + CNN
**输入形式：** CWD 时频图
**监督方式：** 监督分类
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 入门复现、经典 baseline、LPI 雷达波形识别 pipeline

该项目实现了一个比较完整的 LPI 雷达波形识别流程，包括雷达波形生成、CWD 时频分析、时频图转换、CNN 训练与评估。对于刚进入该方向的研究者，它是最适合先跑通的项目之一。

**链接：** [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)

---

#### 4.2.2 AIMC-image-classification

**推荐程度：** ⭐⭐⭐⭐⭐
**方法类型：** Spectrogram image classification
**输入形式：** AIMC-Spec spectrogram
**监督方式：** 监督分类
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 标准 benchmark、模型公平对比、轻量 CNN / ResNet / Transformer 对比

该项目围绕 AIMC-Spec 数据集搭建脉内调制分类框架，适合用于统一输入格式下的模型评估。它比零散自生成数据更适合作为公开 benchmark。

**链接：** [GitHub](https://github.com/seb-cocks/AIMC-image-classification)

---

#### 4.2.3 Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation

**推荐程度：** ⭐⭐⭐⭐
**方法类型：** 脉内调制信号仿真 + 时频图生成 + few-shot / 多模态识别
**输入形式：** 一维 I/Q 序列、WVD、CWD、STFT、CWT 等
**监督方式：** 监督 / 少样本
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 自建数据集、少样本实验、多模态融合、时频变换对比

该项目适合用来构建不同调制类别、不同 SNR、不同表示方式的数据，并可用于比较一维序列模型和二维时频图模型。

**链接：** [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)

---

#### 4.2.4 DNCNet-pytorch

**推荐程度：** ⭐⭐⭐⭐
**方法类型：** Deep radar signal denoising + recognition
**输入形式：** 一维雷达信号
**监督方式：** 监督分类
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 低 SNR 识别、去噪网络、鲁棒性实验

DNCNet 将雷达信号去噪和分类结合起来，适合研究低信噪比条件下分类器性能下降的问题。它可以作为“先增强 / 去噪，再识别”路线的代表性开源实现。

**链接：** [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)

---

#### 4.2.5 MAPNet

**推荐程度：** ⭐⭐⭐⭐
**方法类型：** Multi-scale atrous pyramid network + deep metric learning
**输入形式：** 雷达信号表示
**监督方式：** 监督度量学习
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 鲁棒识别、开放集识别、embedding 学习、度量学习对照

MAPNet 关注雷达信号识别的鲁棒性，通过多尺度空洞金字塔结构和度量学习增强特征表达能力。适合用于研究已知类识别之外的鲁棒判别和开放集扩展。

**链接：** [GitHub](https://github.com/bryantky/MAPNet)

---

#### 4.2.6 CTNet-SSL

**推荐程度：** ⭐⭐⭐
**方法类型：** CNN-Transformer hybrid network + contrastive self-supervised learning
**输入形式：** 雷达信号
**监督方式：** 自监督 / 对比学习 / 少量标注微调
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 标注样本稀缺、自监督预训练、对比学习

该项目面向 radar signal modulation recognition 中训练数据不足的问题，适合用于参考 CNN-Transformer 混合结构和自监督对比学习思路。由于公开文档较少，复现实验前需要仔细检查数据、训练脚本和参数设置。

**链接：** [GitHub](https://github.com/wanan0414/CTNet-SSL)

---

#### 4.2.7 SEMTN

**推荐程度：** ⭐⭐⭐
**方法类型：** Self-enhanced Multidimensional Taylor Network
**输入形式：** `.mat` 雷达数据
**监督方式：** 监督分类
**是否严格符合雷达调制识别定义：** 是
**适合用途：** 多信道、多 SNR、不同衰落环境下的鲁棒识别

SEMTN 仓库中包含 AWGN、Nakagami、Rayleigh、Rician 等不同信道条件下的数据文件和 MATLAB 代码。其 README 较简略，使用前需要自行解析数据格式和训练流程。

**链接：** [GitHub](https://github.com/Guqih/SEMTN)

---

#### 4.2.8 rsrc-for-pub

**推荐程度：** ⭐⭐⭐⭐
**方法类型：** 多个雷达信号识别论文代码与数据集合
**输入形式：** 多种，包括时频图、信号序列、数据集代码
**监督方式：** 多种
**是否严格符合雷达调制识别定义：** 部分子项目符合
**适合用途：** 低 SNR、多分量雷达信号识别、传统 + 深度方法对比

该仓库包含多个雷达信号识别相关子项目，例如 SFUnet-DCNN、SQDandFE 等。其中部分子项目严格属于 LPI 雷达波形识别或雷达信号识别，部分则属于多分量或扩展任务。使用时应逐个子目录判断任务设定。

**链接：** [GitHub](https://github.com/stu-cjlu-sp/rsrc-for-pub)

---

#### 4.2.9 Sig2text

**推荐程度：** ⭐⭐⭐
**方法类型：** Vision-language model / signal-to-text parsing
**输入形式：** 雷达波形时频表示
**监督方式：** 监督解析
**是否严格符合雷达调制识别定义：** 部分符合，属于扩展任务
**适合用途：** 调制识别 + 参数估计 + 信号语义解析

Sig2text 不是传统意义上的单标签调制分类方法，而是将雷达信号识别扩展为文本化 / 符号化解析任务。它适合关注“调制类型 + 参数 + 可解释描述”的研究方向。

**链接：** [Paper](https://arxiv.org/abs/2503.15213)

---

## 5. 推荐实验设置 / Recommended Experimental Setup

如果研究目标是 **雷达调制类型识别（radar modulation type recognition）**，建议采用以下实验设置。

---

### 5.1 主 benchmark

建议优先选择以下数据集作为主 benchmark：

| Goal 目标                               | Recommended Benchmark 推荐数据集 |
| ------------------------------------- | --------------------------- |
| 标准 spectrogram / image classification | AIMC-Spec / AIMC-Spec-v2    |
| 一维 I/Q 序列分类                           | DeepRadar2022               |
| 多任务：调制识别 + 参数估计                       | RadChar                     |
| 少样本 / 自监督                             | RadCharSSL                  |
| 雷达 + 通信联合识别                           | RadarCommDataset            |
| 经典 LPI 波形识别 pipeline                  | LPI-Net generated data      |
| 自建脉内调制仿真数据                            | Liuyh0308 simulation code   |

---

### 5.2 基线方法

建议实现并比较以下 baseline：

| Category 类别                 | Methods 方法                                                               |
| --------------------------- | ------------------------------------------------------------------------ |
| Traditional features        | Instantaneous frequency + SVM, HOC + SVM, handcrafted TFI features       |
| Time-frequency image models | STFT + CNN, CWD + CNN, WVD + ResNet, CWT + DenseNet                      |
| 1D I/Q models               | 1D-CNN, CNN-LSTM, TCN, Transformer encoder                               |
| Robust recognition          | Denoising + classifier, DNCNet-style model, attention CNN                |
| Few-shot / SSL              | Prototypical network, MAML, contrastive learning, masked signal modeling |
| Open-set recognition        | Metric learning, threshold-based rejection, OpenMax, energy score        |
| Multi-task learning         | Classification + parameter regression                                    |

---

### 5.3 建议流程

#### 5.3.1 标准监督分类流程

```text
Raw radar I/Q signal
      ↓
Preprocessing
      ↓
Representation
      ├── I/Q sequence
      ├── STFT spectrogram
      ├── CWD / WVD / CWT time-frequency image
      └── handcrafted features
      ↓
Classifier
      ├── SVM / KNN / Random Forest
      ├── 1D-CNN / LSTM / TCN / Transformer
      └── CNN / ResNet / ViT
      ↓
Modulation type prediction
      ↓
Evaluation: Accuracy / Macro-F1 / Confusion Matrix / Accuracy-SNR curve
```

#### 5.3.2 低信噪比鲁棒识别流程

```text
Noisy radar signal
      ↓
Denoising or enhancement module
      ↓
Signal / time-frequency representation
      ↓
Classifier
      ↓
Accuracy vs. SNR evaluation
```

#### 5.3.3 少样本 / 自监督流程

```text
Unlabeled RF / radar data
      ↓
Self-supervised pretraining
      ↓
Few-shot labeled radar samples
      ↓
Linear probing or fine-tuning
      ↓
N-way K-shot evaluation
```

---

## 6. 相关任务 / Related Tasks

雷达调制类型识别经常与其他任务混在一起，使用资源时需要明确区分。

| Related Task 相关任务                        | Task Definition 任务定义            | 与调制识别的关系        |
| ---------------------------------------- | ------------------------------- | --------------- |
| Communication AMC                        | 识别通信信号调制，如 QPSK、QAM、FSK         | 方法相似，但数据和物理机制不同 |
| Radar pulse deinterleaving               | 将混叠 PDW 脉冲流分到不同辐射源              | 分选任务，不是调制类型分类   |
| Radar emitter identification / SEI       | 识别具体辐射源或设备身份                    | 关注设备指纹，不只是波形类别  |
| Radar target recognition / SAR ATR       | 识别目标类别，如飞机、车辆、舰船                | 目标识别，不是信号调制识别   |
| Radar spectrum detection                 | 判断宽带频谱中是否存在雷达信号                 | 检测任务，可作为识别前处理   |
| Radar activity segmentation              | 对长 I/Q 序列做 sample-wise 雷达活动标注   | 分割任务，不是单信号分类    |
| Radar jamming recognition                | 识别干扰样式，如 ISRJ、VGPO、comb jamming | 对象是干扰信号，和调制识别相邻 |
| Multi-component radar signal recognition | 一个观测中有多个雷达信号成分，需要识别多个类别         | 是调制识别的复杂扩展      |
| Signal parameter estimation              | 估计带宽、脉宽、调频斜率、码长等参数              | 可与调制识别组成多任务学习   |
| Signal-to-text / symbolic parsing        | 将雷达信号解析为文本或结构化参数                | 是调制识别的上位任务      |

---

## 7. 推荐阅读与入门资源 / Recommended Reading and Starting Points

### 标准调制识别 benchmark

* [AIMC-Spec paper](https://arxiv.org/html/2601.08265v2)
* [AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification)
* [AIMC-Spec-v2 Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)
* [DeepRadar2022](https://www.kaggle.com/datasets/khilian/deepradar)

### 经典 LPI 雷达波形识别

* [LPI-Radar-Waveform-Recognition](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)
* [MathWorks LPI Radar Waveform Classification Example](https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html)

### 数据集与少样本 / 自监督方向

* [RadChar](https://github.com/abcxyzi/RadChar)
* [RadCharSSL](https://github.com/abcxyzi/RadCharSSL)
* [RadCharSSL Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)
* [RadarCommDataset](https://github.com/ANDROComputationalSolutions/RadarCommDataset)

### 低 SNR 与鲁棒识别

* [DNCNet-pytorch](https://github.com/dumingyang20/DNCNet-pytorch)
* [MAPNet](https://github.com/bryantky/MAPNet)
* [SEMTN](https://github.com/Guqih/SEMTN)
* [rsrc-for-pub](https://github.com/stu-cjlu-sp/rsrc-for-pub)

### 数据生成与仿真

* [Radar-Intra-Pulse-Modulation-Signal-Simulation](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)
* [Radar-Intra-Pulse-Modulation-Simulation](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)

### 扩展任务

* [Sig2text paper](https://arxiv.org/abs/2503.15213)
* [RadDet](https://github.com/abcxyzi/RadDet)
* [RadSeg](https://github.com/abcxyzi/radseg)

---

## 8. 说明 / Notes

* 与通信 automatic modulation classification 相比，雷达调制类型识别的数据集和开源代码明显更少；
* 很多论文使用自生成数据，实验结果高度依赖调制参数、SNR 范围、采样率、脉宽、带宽、训练测试划分；
* 报告结果时应明确说明输入是 I/Q 序列、频谱图、STFT、CWD、WVD、CWT 还是其他表示；
* 如果使用合成数据，应公布信号参数范围、噪声模型、信道模型、SNR 设置和随机种子；
* 如果研究低 SNR 鲁棒性，应报告 accuracy-SNR curve，而不是只报告平均准确率；
* 如果研究少样本，应明确 N-way K-shot 设置、base / novel 类划分、episode 数量和置信区间；
* 如果研究开放集，应明确 known / unknown 类划分，并报告 AUROC、OSCR、FPR95 等指标；
* 如果使用 RadarCommDataset 或 RadioML，应明确哪些实验是雷达调制识别，哪些是通信 AMC 或雷达通信联合识别；
* 不应将雷达检测、脉冲分选、辐射源个体识别、目标识别、干扰识别直接写成雷达调制类型识别；
* 本仓库中的推荐程度会随着项目更新、数据开放情况和复现记录变化而调整。

---

## 9. 引用与贡献 / Citation and Contribution

本资源列表由 **厦门大学信息学院 SMARTDSP 实验室** 整理与维护，旨在汇总雷达调制类型识别（Radar Modulation Type Recognition / Radar Waveform Recognition / Automatic Intrapulse Modulation Classification）相关的公开数据集、开源代码、代表性方法和可复现实验资源。

如果本仓库对你的学习、研究或项目开发有帮助，欢迎在论文、报告或项目文档中引用本仓库，并请同时引用相关方法、数据集和代码仓库的原始论文或项目页面。

### 维护说明

* 本仓库主要关注雷达调制类型识别、LPI radar waveform recognition、automatic intrapulse modulation classification、radar signal recognition 及其相关任务；
* 资源整理过程中会尽量标注任务类型、数据可用性、监督方式和是否严格符合雷达调制类型识别定义；
* 由于部分开源项目的数据说明、许可证、复现流程可能不完整，相关判断会随着项目更新持续修正；
* 本仓库中的推荐程度仅代表整理者基于任务匹配度、开源程度、复现价值和数据说明完整度给出的参考意见。

### 贡献方式

欢迎通过 Issue 或 Pull Request 补充和修正资源，包括但不限于：

* 新发布的雷达调制类型识别数据集；
* LPI 雷达波形识别、AIMC、radar signal recognition 相关开源实现；
* 可复现的 benchmark 结果；
* 数据集可用性、链接失效或许可证信息的修正；
* 关于某个方法是否严格属于雷达调制类型识别的补充说明；
* 通信 AMC、雷达检测、分选、目标识别等相关但不同任务的边界说明。

建议提交资源时尽量包含以下信息：

```text
项目名称：
项目链接：
任务类型：
方法类型：
输入表示：
是否开源代码：
是否开源数据：
是否包含标签：
是否严格属于雷达调制类型识别：
推荐理由：
备注：
```
