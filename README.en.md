# Radar Modulation Type Recognition Resources

<p align="center">
  <a href="./README.md">简体中文</a> |
  <b>English</b>
</p>

This repository continuously collects open-source resources related to **Radar Modulation Type Recognition / Radar Waveform Recognition / Automatic Intrapulse Modulation Classification (AIMC)**. It focuses on task definitions, public datasets, open-source implementations, representative methods, evaluation metrics, reproducible experimental settings, and task boundaries.

Radar modulation type recognition aims to identify the waveform or intrapulse modulation type of an intercepted radar signal, usually represented as a complex baseband **I/Q sequence**, time-frequency image, spectrogram, spectrum, or other signal representation. Common radar waveform or modulation classes include **LFM, NLFM, SFM, Costas, Barker, Frank, P1/P2/P3/P4, BPSK, QPSK, FSK, OFDM, FMCW**, etc.

Open-source resources in this field are relatively scattered. Different projects may use terms such as `radar waveform recognition`, `LPI radar recognition`, `radar signal recognition`, `automatic intrapulse modulation classification`, and `radar modulation classification` in slightly different ways. Therefore, this repository attempts to manually organize related resources and annotate their task type, data availability, supervision setting, whether they strictly match the definition of radar modulation type recognition, and reproducibility.

---

## Highlights

* **Recommended standard benchmarks:** AIMC-Spec, DeepRadar2022, RadChar / RadCharSSL.
* **Classic reproduction starting point:** LPI-Radar-Waveform-Recognition, based on CWD time-frequency analysis + CNN / LPI-Net.
* **Data generation starting points:** Radar-Intra-Pulse-Modulation-Signal-Simulation and Radar-Intra-Pulse-Modulation-Simulation.
* **Low-SNR robust recognition:** DNCNet, SEMTN, SFUnet-DCNN and related methods are worth attention.
* **Few-shot / self-supervised learning:** RadCharSSL, CTNet-SSL, and few-shot radar signal recognition resources.
* **Extended tasks:** Multi-component radar signal recognition, parameter estimation, signal parsing, and radar-communication joint recognition are closely related, but should not be confused with standard single-signal modulation classification.
* **Important distinction:** Communication AMC, radar target recognition, radar detection, pulse deinterleaving, specific emitter identification, and RF fingerprinting are not the same task as radar modulation type recognition.

> Note: The ⭐ recommendation level in this repository is a subjective rating based on task relevance, openness, reproducibility, and completeness of data documentation. It does not represent GitHub stars.

---

## Table of Contents

* [1. Overview](#1-overview)

  * [1.1 What is Radar Modulation Type Recognition?](#11-what-is-radar-modulation-type-recognition)
  * [1.2 Common Input Representations](#12-common-input-representations)
  * [1.3 Common Modulation Classes](#13-common-modulation-classes)
  * [1.4 Common Evaluation Metrics](#14-common-evaluation-metrics)
* [2. Method Taxonomy and Task Fit](#2-method-taxonomy-and-task-fit)

  * [2.1 Traditional Feature + Classifier Methods](#21-traditional-feature--classifier-methods)
  * [2.2 Time-Frequency Image + Deep Learning Methods](#22-time-frequency-image--deep-learning-methods)
  * [2.3 End-to-End 1D I/Q Sequence Methods](#23-end-to-end-1d-iq-sequence-methods)
  * [2.4 Denoising / Enhancement + Recognition Methods](#24-denoising--enhancement--recognition-methods)
  * [2.5 Few-Shot, Self-Supervised, and Domain Adaptation Methods](#25-few-shot-self-supervised-and-domain-adaptation-methods)
  * [2.6 Metric Learning, Open-Set, and Unknown-Class Recognition](#26-metric-learning-open-set-and-unknown-class-recognition)
  * [2.7 Which Methods Strictly Match the Definition?](#27-which-methods-strictly-match-the-definition)
* [3. Datasets](#3-datasets)

  * [3.1 Dataset Overview](#31-dataset-overview)
  * [3.2 Key Dataset Notes](#32-key-dataset-notes)
* [4. Methods and Implementations](#4-methods-and-implementations)

  * [4.1 Method and Code Overview](#41-method-and-code-overview)
  * [4.2 Key Method Notes](#42-key-method-notes)
* [5. Recommended Experimental Setup](#5-recommended-experimental-setup)

  * [5.1 Main Benchmarks](#51-main-benchmarks)
  * [5.2 Baseline Methods](#52-baseline-methods)
  * [5.3 Suggested Pipelines](#53-suggested-pipelines)
* [6. Related Tasks](#6-related-tasks)
* [7. Recommended Reading and Starting Points](#7-recommended-reading-and-starting-points)
* [8. Notes](#8-notes)
* [9. Citation and Contribution](#9-citation-and-contribution)

---

## 1. Overview

### 1.1 What is Radar Modulation Type Recognition?

Radar modulation type recognition is also commonly referred to as **radar waveform recognition**, **LPI radar waveform recognition**, **radar signal modulation recognition**, or **Automatic Intrapulse Modulation Classification (AIMC)**.

Given an intercepted radar signal:

```text
x = {x1, x2, ..., xN}, xi ∈ C
```

where `x` is usually a complex baseband I/Q sampling sequence, the goal is to predict its corresponding modulation or waveform class:

```text
y ∈ {LFM, NLFM, SFM, Costas, Barker, Frank, P1, P2, P3, P4, BPSK, QPSK, FSK, FMCW, ...}
```

In the most common research setting, radar modulation type recognition is a **supervised multi-class classification problem**. During training, labeled radar modulation data are used. During testing, the model predicts the modulation type of an unknown signal.

However, in more complex electronic reconnaissance and electronic support scenarios, the task may be extended to:

* robust modulation recognition under low SNR;
* few-shot modulation recognition;
* open-set recognition of unknown modulation types;
* multi-component / overlapping radar signal recognition;
* modulation type recognition + parameter estimation;
* semantic radar signal parsing;
* joint recognition of radar and communication signals.

It should be noted that radar modulation type recognition is **not** radar target recognition, and it is also **not** radar pulse deinterleaving. The former identifies waveform or intrapulse modulation classes, while the latter usually focuses on target classes, emitter clusters, pulse ownership, or device identity.

---

### 1.2 Common Input Representations

| Input Representation      | Description                                                  | Usage                                                                   |
| ------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| I/Q sequence              | Complex baseband in-phase / quadrature sampling sequence     | The most direct input form, suitable for 1D-CNN, LSTM, Transformer      |
| Amplitude / phase         | Amplitude, phase, or instantaneous phase                     | Useful for traditional features and phase-coded waveform recognition    |
| Instantaneous frequency   | Instantaneous frequency curve                                | Helpful for LFM, NLFM, SFM, FSK and other frequency-modulated waveforms |
| Spectrum                  | Spectrum or power spectrum                                   | Suitable for coarse waveform recognition                                |
| Spectrogram               | STFT spectrogram                                             | Commonly used input for image classification methods                    |
| Time-frequency image, TFI | WVD, CWD, CWT, FSST and other time-frequency representations | Very common in LPI radar recognition                                    |
| Ambiguity function        | Time-delay and Doppler representation                        | Reflects the delay-Doppler structure of radar waveforms                 |
| Multi-modal features      | I/Q + time-frequency image + statistical features            | Suitable for multi-modal fusion and few-shot learning                   |

---

### 1.3 Common Modulation Classes

Different datasets and papers use different class sets. Common classes include:

| Family                      | Examples                                            |
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

### 1.4 Common Evaluation Metrics

| Metric                     | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| Accuracy                   | Overall classification accuracy, the most commonly used metric |
| Macro-F1                   | More informative when classes are imbalanced                   |
| Precision / Recall         | Used for class-wise false alarm and missed detection analysis  |
| Confusion matrix           | Used to analyze which modulation types are easily confused     |
| Accuracy vs. SNR           | Core metric for low-SNR robustness analysis                    |
| Open-set AUROC             | Common metric for open-set / unknown-class recognition         |
| Few-shot accuracy          | Recognition accuracy under N-way K-shot settings               |
| Cross-dataset accuracy     | Measures cross-dataset generalization ability                  |
| Parameter estimation error | Used when the task includes parameter estimation               |

---

## 2. Method Taxonomy and Task Fit

Radar modulation type recognition methods can be roughly divided into **traditional feature + classifier methods, time-frequency image + deep learning methods, end-to-end 1D I/Q models, denoising / enhancement + recognition methods, few-shot / self-supervised / domain adaptation methods, metric learning / open-set recognition methods, and multi-task parsing methods**.

---

### 2.1 Traditional Feature + Classifier Methods

These methods usually extract manually designed features from I/Q signals and then use traditional machine learning classifiers.

| Method                           | Main Idea                                                                                        | Advantages                     | Limitations                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------ | -------------------------------------------- |
| Instantaneous feature + SVM      | Extract instantaneous frequency, instantaneous phase, envelope and other features, then classify | Interpretable                  | Sensitive to noise and parameter variation   |
| Higher-order cumulants           | Use higher-order statistical features to distinguish modulation types                            | Effective for some modulations | Limited coverage for complex radar waveforms |
| Time-frequency feature + KNN/SVM | Extract texture or shape features from STFT/WVD/CWD images                                       | Suitable for LPI waveforms     | Feature design depends on experience         |
| Sparse representation            | Represent different waveforms using sparse dictionaries                                          | Theoretically clear            | Usually computationally expensive            |
| Template matching                | Match with known modulation templates or parameter templates                                     | Suitable for known waveforms   | Limited generalization                       |

Traditional methods are useful as interpretable baselines, but they are usually less robust than deep learning methods under low SNR, parameter variation, unknown channels, multipath, and cross-dataset settings.

---

### 2.2 Time-Frequency Image + Deep Learning Methods

This is currently one of the most common deep learning routes for radar modulation recognition. A typical pipeline is:

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

| Method Family                    | Main Idea                                                          | Notes                                                    |
| -------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- |
| STFT + CNN                       | Convert signals into spectrograms and perform image classification | Simple and easy to reproduce                             |
| CWD + CNN                        | Use Choi-Williams Distribution to obtain time-frequency images     | Representative route of LPI-Net                          |
| WVD / PWVD + CNN                 | Use Wigner-Ville type time-frequency distributions                 | High resolution, but cross-term interference is an issue |
| CWT + CNN                        | Use continuous wavelet transform for multi-scale representation    | Friendly to non-stationary signals                       |
| FSST / synchrosqueezing + CNN    | Use finer time-frequency reassignment                              | More complex preprocessing                               |
| ResNet / DenseNet / EfficientNet | Use mature image classification networks                           | Easy to build strong baselines                           |
| ViT / Swin / CNN-Transformer     | Use attention to model global structures                           | Requires more data and stronger regularization           |

---

### 2.3 End-to-End 1D I/Q Sequence Methods

These methods directly use I/Q sequences as input and avoid time-frequency image preprocessing.

| Method                        | Main Idea                                                            | Notes                                                             |
| ----------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 1D-CNN                        | Apply convolutional feature extraction to two-channel I/Q sequences  | Simple and efficient                                              |
| CNN-LSTM / CNN-GRU            | CNN extracts local features, RNN models temporal dependencies        | Suitable for signals with clear temporal structure                |
| TCN                           | Use temporal convolutional networks to model long-range dependencies | Easier to parallelize than RNNs                                   |
| Transformer encoder           | Use self-attention to model global sequence relationships            | Requires sufficient data and regularization                       |
| Complex-valued neural network | Directly model complex-valued signals                                | More physically consistent with I/Q data, but harder to implement |

The advantage of 1D methods is that the input is closer to the raw signal. Their disadvantages include weaker interpretability and sensitivity to sampling rate, pulse width, bandwidth, carrier frequency offset, and other factors.

---

### 2.4 Denoising / Enhancement + Recognition Methods

Low SNR is one of the key challenges in radar modulation recognition. Common solutions include:

| Method                         | Main Idea                                                           | Representative Direction |
| ------------------------------ | ------------------------------------------------------------------- | ------------------------ |
| Denoising + classifier         | Denoise first, then classify                                        | DNCNet                   |
| Time-frequency enhancement     | Denoise, enhance, or reconstruct time-frequency images              | SFUnet-DCNN, TFGM, etc.  |
| Residual / attention network   | Use residual structures and channel attention to improve robustness | ResNet, SE, CBAM, MAPNet |
| Multi-scale feature extraction | Capture modulation textures at multiple scales                      | MAPNet, pyramid networks |
| Data augmentation              | Add noise, frequency offset, time shift, amplitude perturbation     | Improve generalization   |

---

### 2.5 Few-Shot, Self-Supervised, and Domain Adaptation Methods

Real radar data are difficult to obtain and expensive to label. Therefore, few-shot and self-supervised learning are becoming increasingly important.

| Method                                       | Main Idea                                                                     | Supervision Level                                |
| -------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| Few-shot learning                            | Recognize new modulation classes under N-way K-shot settings                  | Few labeled samples                              |
| MAML / meta-learning                         | Learn an initialization that can quickly adapt to new tasks                   | Few-shot supervised                              |
| Contrastive learning                         | Construct positive and negative pairs to learn discriminative representations | Self-supervised / weakly supervised / supervised |
| Masked signal modeling                       | Mask part of I/Q or time-frequency images and reconstruct them                | Self-supervised                                  |
| Domain adaptation                            | Transfer from communication or other RF data to the radar domain              | Unsupervised / semi-supervised / few-shot        |
| Self-supervised pretraining + linear probing | Pretrain representations first, then fine-tune with a small amount of labels  | Self-supervised + few labeled samples            |

---

### 2.6 Metric Learning, Open-Set, and Unknown-Class Recognition

In electronic reconnaissance, modulation types not seen during training may appear during testing. Therefore, open-set recognition and metric learning are also important.

| Method                          | Main Idea                                                                       | Notes                                |
| ------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------ |
| Triplet loss                    | Pull samples from the same class closer and push different classes apart        | Requires labels during training      |
| Contrastive loss                | Learn similar / dissimilar signal pairs                                         | Can be supervised or self-supervised |
| Prototypical network            | Learn a prototype center for each class                                         | Common in few-shot learning          |
| Metric learning + clustering    | Learn embeddings and then perform clustering or nearest-neighbor classification | Suitable for new-class extension     |
| OpenMax / thresholding          | Reject unknown classes using confidence or distance thresholds                  | Simple open-set baseline             |
| Energy-based open-set detection | Detect unknown samples using energy scores                                      | Deep open-set direction              |

---

### 2.7 Which Methods Strictly Match the Definition?

> **Strict definition:** Given a single radar signal or single-pulse I/Q / time-frequency input, output its radar modulation or waveform class.

| Method Family                            | Matches the Task?          | Standard Classification?                 | Recommendation | Comment                                                        |
| ---------------------------------------- | -------------------------- | ---------------------------------------- | -------------- | -------------------------------------------------------------- |
| STFT/CWD/WVD/CWT + CNN                   | Yes                        | Yes                                      | ⭐⭐⭐⭐⭐          | Currently common and easy to reproduce                         |
| LPI-Net / CWD-TFA + CNN                  | Yes                        | Yes                                      | ⭐⭐⭐⭐⭐          | Classic LPI radar waveform recognition route                   |
| AIMC-Spec image classification           | Yes                        | Yes                                      | ⭐⭐⭐⭐⭐          | Recommended spectrogram benchmark                              |
| 1D-CNN / LSTM / Transformer on I/Q       | Yes                        | Yes                                      | ⭐⭐⭐⭐           | Suitable for end-to-end baselines                              |
| Denoising + classification               | Yes                        | Yes                                      | ⭐⭐⭐⭐           | Suitable for low-SNR scenarios                                 |
| Few-shot / SSL radar recognition         | Yes                        | Yes / few-shot                           | ⭐⭐⭐⭐           | Suitable for limited-label scenarios                           |
| Metric learning / open-set recognition   | Yes                        | Extended classification                  | ⭐⭐⭐⭐           | Suitable for unknown-class and robust recognition              |
| Multi-component radar signal recognition | Partially                  | Multi-label / multi-instance             | ⭐⭐⭐            | More difficult than standard single-signal classification      |
| Parameter estimation / Sig2text          | Partially                  | Multi-task / parsing                     | ⭐⭐⭐            | Extended task of modulation recognition                        |
| RadarComm joint classification           | Partially                  | Mixed radar-communication classification | ⭐⭐⭐            | Radar and communication labels should be distinguished         |
| Communication AMC                        | Related but not equivalent | Communication modulation classification  | ⭐⭐             | Should not be directly treated as radar modulation recognition |
| Radar pulse deinterleaving               | Different task             | Clustering / sorting                     | ⭐⭐             | Identifies pulse ownership, not modulation type                |
| Radar target recognition / SAR ATR       | Different task             | Target classification                    | ⭐              | Identifies targets, not waveform modulation                    |

---

## 3. Datasets

This section collects datasets, simulation data, data generation code, and related-task data for radar modulation type recognition. The recommendation level is mainly based on **task relevance, openness, label clarity, benchmark value, documentation completeness, and reproducibility**.

---

### 3.1 Dataset Overview

| Dataset                                 | Task Fit                                               | Data Type                            | Labels                                          | Open Source?                    | Recommendation | Links                                                                                                                                                                                   | Notes                                                                                  |
| --------------------------------------- | ------------------------------------------------------ | ------------------------------------ | ----------------------------------------------- | ------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| AIMC-Spec / AIMC-Spec-v2                | Standard intrapulse modulation classification          | Spectrogram / image                  | Modulation labels                               | Yes                             | ⭐⭐⭐⭐⭐          | [Paper](https://arxiv.org/html/2601.08265v2) / [GitHub](https://github.com/seb-cocks/AIMC-image-classification) / [Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2) | Class numbers may differ between paper and code versions; specify the version used     |
| DeepRadar2022                           | Radar modulation classification                        | I/Q sequence                         | 23 radar modulation classes                     | Yes                             | ⭐⭐⭐⭐⭐          | [Kaggle](https://www.kaggle.com/datasets/khilian/deepradar)                                                                                                                             | Suitable for 1D I/Q sequence classification                                            |
| RadChar                                 | Radar signal characterization / multi-task recognition | I/Q sequence, HDF5                   | Modulation labels + parameter labels            | Yes                             | ⭐⭐⭐⭐           | [GitHub](https://github.com/abcxyzi/RadChar)                                                                                                                                            | Not only classification; also supports parameter regression                            |
| RadCharSSL                              | Few-shot / self-supervised radar recognition           | I/Q datasets                         | Few-shot / SSL settings                         | Yes                             | ⭐⭐⭐⭐           | [GitHub](https://github.com/abcxyzi/RadCharSSL) / [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)                                                                | Suitable for self-supervised pretraining and few-shot experiments                      |
| RadarCommDataset                        | Joint radar + communication recognition                | I/Q waveform                         | Radar / communication / modulation multi-labels | Yes                             | ⭐⭐⭐            | [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset)                                                                                                               | Mixed dataset, not pure radar modulation classification                                |
| LPI-Net generated data                  | LPI radar waveform recognition                         | Synthetic radar waveform + CWD TFI   | LPI waveform labels                             | Generatable                     | ⭐⭐⭐⭐           | [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)                                                                                                               | Suitable for reproducing the CWD + CNN pipeline                                        |
| Liuyh0308 simulation data               | Radar intrapulse modulation recognition                | `.mat` I/Q + WVD/CWD/STFT/CWT images | Modulation labels                               | Generatable                     | ⭐⭐⭐⭐           | [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)                                                                                                   | Suitable for few-shot and multi-modal fusion experiments                               |
| DNCNet SIGNAL-8                         | Low-SNR radar signal recognition                       | 1D radar signals                     | 8 radar signal classes                          | Provided in repo / reproducible | ⭐⭐⭐⭐           | [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)                                                                                                                                | Suitable for denoising + recognition tasks                                             |
| SEMTN data                              | Multi-channel / multi-SNR radar modulation recognition | `.mat` data                          | Modulation class + SNR / channel                | Provided in repo                | ⭐⭐⭐            | [GitHub](https://github.com/Guqih/SEMTN)                                                                                                                                                | README is brief; users need to inspect data fields                                     |
| MAPNet data                             | Robust radar signal recognition                        | `.npy` / model weights               | Modulation labels                               | Partially public                | ⭐⭐⭐            | [GitHub](https://github.com/bryantky/MAPNet)                                                                                                                                            | Data completeness should be checked before reproduction                                |
| Radar-Intra-Pulse-Modulation-Simulation | Data generation                                        | Python synthetic signals             | Labels can be generated                         | Code available                  | ⭐⭐⭐            | [GitHub](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)                                                                                                       | Suitable for teaching and supplementary simulation                                     |
| RadioML / DeepSig                       | Communication AMC                                      | Communication I/Q                    | Communication modulation labels                 | Yes                             | ⭐⭐             | [RadioML example](https://www.kaggle.com/datasets/pinxau1000/radioml2018)                                                                                                               | Not radar modulation recognition; can only be used for transfer learning or comparison |
| RadDet                                  | Radar spectrum detection                               | Wideband spectrum                    | Detection labels                                | Yes                             | ⭐⭐             | [GitHub](https://github.com/abcxyzi/RadDet)                                                                                                                                             | Related task, not modulation classification                                            |
| RadSeg                                  | Radar activity segmentation                            | Long I/Q sequence                    | Sample-wise labels                              | Yes                             | ⭐⭐             | [GitHub](https://github.com/abcxyzi/radseg)                                                                                                                                             | Related task, not standard modulation recognition                                      |

---

### 3.2 Key Dataset Notes

#### 3.2.1 AIMC-Spec / AIMC-Spec-v2

**Recommendation:** ⭐⭐⭐⭐⭐
**Suitable for:** standard intrapulse modulation classification benchmark, spectrogram image classification, unified model evaluation
**Data type:** spectrogram / time-frequency image of radar intrapulse modulation signals
**Labels:** modulation class labels
**Note:** The number of classes may differ between the paper version and the GitHub / Kaggle version. Please specify the exact version used in experiments.

AIMC-Spec is a highly suitable standardized benchmark for automatic intrapulse modulation classification. It formulates radar modulation recognition as a spectrogram-based image classification problem and is suitable for reproducing CNN, ResNet, lightweight networks, denoising networks, and Transformer models.

**Links:** [Paper](https://arxiv.org/html/2601.08265v2) / [GitHub](https://github.com/seb-cocks/AIMC-image-classification) / [Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)

---

#### 3.2.2 DeepRadar2022

**Recommendation:** ⭐⭐⭐⭐⭐
**Suitable for:** 1D I/Q radar modulation classification, LSTM / CNN / Transformer baselines
**Data type:** 1024 × 2 I/Q sequence
**Labels:** 23 radar modulation classes
**Note:** Suitable for direct 1D model experiments without converting signals into images.

DeepRadar2022 is a commonly used radar modulation classification dataset covering multiple radar modulation types. It is suitable for end-to-end I/Q sequence recognition experiments.

**Link:** [Kaggle](https://www.kaggle.com/datasets/khilian/deepradar)

---

#### 3.2.3 RadChar

**Recommendation:** ⭐⭐⭐⭐
**Suitable for:** radar signal characterization, multi-task learning, classification + parameter estimation
**Data type:** synthetic radar I/Q data
**Labels:** classification and regression labels
**Note:** It is not just a modulation classification dataset; it is more suitable for radar signal characterization.

The value of RadChar lies in the fact that it provides both modulation class labels and parameter estimation labels, making it suitable for multi-task models involving both modulation recognition and parameter estimation.

**Link:** [GitHub](https://github.com/abcxyzi/RadChar)

---

#### 3.2.4 RadCharSSL

**Recommendation:** ⭐⭐⭐⭐
**Suitable for:** few-shot radar signal recognition, self-supervised pretraining, RF domain adaptation
**Data type:** radar I/Q data and few-shot splits
**Labels:** supports few-shot / SSL experimental settings
**Note:** Suitable for limited-label scenarios; not necessarily the only benchmark for ordinary supervised classification.

RadCharSSL targets few-shot radar signal recognition and self-supervised learning. It is suitable for masked signal modeling, domain adaptation, linear probing, and few-shot fine-tuning.

**Links:** [GitHub](https://github.com/abcxyzi/RadCharSSL) / [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)

---

#### 3.2.5 RadarCommDataset

**Recommendation:** ⭐⭐⭐
**Suitable for:** joint radar + communication recognition, ISAC / spectrum coexistence, cross-domain learning
**Data type:** radar and communication I/Q waveforms
**Labels:** multi-task labels, including signal type and modulation type
**Note:** This is not a pure radar modulation recognition dataset. Radar waveform and communication modulation should be distinguished.

RadarCommDataset contains both radar and communication waveforms. It is suitable for joint classification, domain adaptation, and cross-task transfer, but should not be directly used as a pure radar modulation recognition benchmark.

**Link:** [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset)

---

#### 3.2.6 LPI-Net Generated Data

**Recommendation:** ⭐⭐⭐⭐
**Suitable for:** LPI radar waveform recognition, CWD time-frequency image, CNN classification
**Data type:** synthetic radar waveform + CWD time-frequency image
**Labels:** LPI waveform labels
**Note:** More suitable for reproducing a classic pipeline than as a large-scale standard benchmark.

The value of this project is its complete workflow: radar signal generation, CWD time-frequency analysis, image conversion, CNN training, and testing.

**Link:** [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)

---

#### 3.2.7 Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation

**Recommendation:** ⭐⭐⭐⭐
**Suitable for:** intrapulse modulation signal simulation, few-shot learning, multi-modal fusion, time-frequency image generation
**Data type:** `.mat` sequence + WVD / CWD / STFT / CWT images
**Labels:** modulation class labels can be generated
**Note:** Data are mainly generated by code. Parameter settings should be checked before reproduction.

This project is suitable for constructing custom radar intrapulse modulation recognition datasets and comparing 1D sequence models, 2D time-frequency image models, and multi-modal fusion methods.

**Link:** [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)

---

## 4. Methods and Implementations

This section collects open-source methods, code implementations, and reproducible experimental frameworks related to radar modulation type recognition. The recommendation level is mainly based on **task relevance, openness, reproducibility, suitability as a baseline, representativeness, and completeness of data documentation**.

---

### 4.1 Method and Code Overview

| Project / Method                                         | Method Type                                   | Input                         | Supervision                            | Strictly RMR? | Recommendation | Links                                                                                                              | Notes                                                                                  |
| -------------------------------------------------------- | --------------------------------------------- | ----------------------------- | -------------------------------------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| LPI-Radar-Waveform-Recognition / LPI-Net                 | CWD + CNN                                     | Time-frequency image          | Supervised                             | Yes           | ⭐⭐⭐⭐⭐          | [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)                                          | Classic entry-level reproduction project                                               |
| AIMC-image-classification                                | Spectrogram + deep models                     | Spectrogram image             | Supervised                             | Yes           | ⭐⭐⭐⭐⭐          | [GitHub](https://github.com/seb-cocks/AIMC-image-classification)                                                   | Official / companion framework for AIMC-Spec                                           |
| Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation | Signal simulation + TFI + few-shot            | I/Q + time-frequency images   | Supervised / few-shot                  | Yes           | ⭐⭐⭐⭐           | [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)                              | Valuable for data generation and multi-modal experiments                               |
| DNCNet-pytorch                                           | Denoising + classification                    | 1D radar signal               | Supervised                             | Yes           | ⭐⭐⭐⭐           | [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)                                                           | Low-SNR robust recognition                                                             |
| MAPNet                                                   | Multi-scale atrous pyramid + metric learning  | Radar signal representation   | Supervised metric learning             | Yes           | ⭐⭐⭐⭐           | [GitHub](https://github.com/bryantky/MAPNet)                                                                       | Suitable for robust recognition and open-set directions                                |
| CTNet-SSL                                                | CNN-Transformer + contrastive SSL             | Radar signal                  | Self-supervised / contrastive learning | Yes           | ⭐⭐⭐            | [GitHub](https://github.com/wanan0414/CTNet-SSL)                                                                   | Documentation is limited; reproduction details need checking                           |
| SEMTN                                                    | Self-enhanced Multidimensional Taylor Network | `.mat` radar data             | Supervised                             | Yes           | ⭐⭐⭐            | [GitHub](https://github.com/Guqih/SEMTN)                                                                           | Multi-channel, multi-SNR data, but README is brief                                     |
| rsrc-for-pub                                             | Multiple radar recognition resources          | TFI / signal / datasets       | Various                                | Partially yes | ⭐⭐⭐⭐           | [GitHub](https://github.com/stu-cjlu-sp/rsrc-for-pub)                                                              | Includes SFUnet-DCNN, SQDandFE and other subprojects                                   |
| RadarCNN                                                 | CNN on DeepRadar2022                          | I/Q or image                  | Supervised                             | Yes           | ⭐⭐             | [GitHub](https://github.com/blackmirag3/RadarCNN)                                                                  | Hobby / demo project, useful for quick reference                                       |
| Radar-Intra-Pulse-Modulation-Simulation                  | Signal simulation                             | I/Q                           | Data generation                        | Partially     | ⭐⭐⭐            | [GitHub](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)                                  | Generates signals only; not a full recognition framework                               |
| Sig2text                                                 | Vision-language / signal-to-text              | Time-frequency representation | Supervised / parsing                   | Extended task | ⭐⭐⭐            | [Paper](https://arxiv.org/abs/2503.15213)                                                                          | Modulation recognition + parameter estimation + symbolic parsing                       |
| MathWorks LPI radar waveform classification              | Time-frequency CNN example                    | Synthetic waveform + TFI      | Supervised                             | Yes           | ⭐⭐⭐            | [Example](https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html) | Good educational value, but not a normal GitHub open-source project                    |
| RadioML / communication AMC repos                        | Communication AMC                             | Communication I/Q             | Supervised                             | No            | ⭐⭐             | [Example](https://github.com/topics/modulation-classification)                                                     | Can be used only for transfer learning or comparison; not radar modulation recognition |

---

### 4.2 Key Method Notes

#### 4.2.1 LPI-Radar-Waveform-Recognition / LPI-Net

**Recommendation:** ⭐⭐⭐⭐⭐
**Method type:** CWD time-frequency analysis + CNN
**Input:** CWD time-frequency image
**Supervision:** supervised classification
**Strictly matches radar modulation recognition:** yes
**Suitable for:** entry-level reproduction, classic baseline, LPI radar waveform recognition pipeline

This project implements a relatively complete LPI radar waveform recognition workflow, including radar waveform generation, CWD time-frequency analysis, time-frequency image conversion, CNN training, and evaluation. For researchers new to this field, it is one of the best projects to run first.

**Link:** [GitHub](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)

---

#### 4.2.2 AIMC-image-classification

**Recommendation:** ⭐⭐⭐⭐⭐
**Method type:** spectrogram image classification
**Input:** AIMC-Spec spectrogram
**Supervision:** supervised classification
**Strictly matches radar modulation recognition:** yes
**Suitable for:** standard benchmark, fair model comparison, lightweight CNN / ResNet / Transformer comparison

This project builds an intrapulse modulation classification framework around the AIMC-Spec dataset. It is suitable for model evaluation under a unified input format and is more suitable as a public benchmark than scattered self-generated datasets.

**Link:** [GitHub](https://github.com/seb-cocks/AIMC-image-classification)

---

#### 4.2.3 Liuyh0308 Radar-Intra-Pulse-Modulation-Signal-Simulation

**Recommendation:** ⭐⭐⭐⭐
**Method type:** intrapulse modulation signal simulation + time-frequency image generation + few-shot / multi-modal recognition
**Input:** 1D I/Q sequence, WVD, CWD, STFT, CWT, etc.
**Supervision:** supervised / few-shot
**Strictly matches radar modulation recognition:** yes
**Suitable for:** custom dataset construction, few-shot experiments, multi-modal fusion, time-frequency transform comparison

This project is suitable for generating data with different modulation classes, SNRs, and representations. It can be used to compare 1D sequence models and 2D time-frequency image models.

**Link:** [GitHub](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)

---

#### 4.2.4 DNCNet-pytorch

**Recommendation:** ⭐⭐⭐⭐
**Method type:** deep radar signal denoising + recognition
**Input:** 1D radar signal
**Supervision:** supervised classification
**Strictly matches radar modulation recognition:** yes
**Suitable for:** low-SNR recognition, denoising networks, robustness experiments

DNCNet combines radar signal denoising with classification and is suitable for studying performance degradation under low SNR. It can be used as a representative implementation of the “enhance / denoise first, then recognize” route.

**Link:** [GitHub](https://github.com/dumingyang20/DNCNet-pytorch)

---

#### 4.2.5 MAPNet

**Recommendation:** ⭐⭐⭐⭐
**Method type:** multi-scale atrous pyramid network + deep metric learning
**Input:** radar signal representation
**Supervision:** supervised metric learning
**Strictly matches radar modulation recognition:** yes
**Suitable for:** robust recognition, open-set recognition, embedding learning, metric learning comparison

MAPNet focuses on robust radar signal recognition and uses a multi-scale atrous pyramid structure and metric learning to improve feature representation. It is suitable for studying robust discrimination and open-set extensions beyond known-class recognition.

**Link:** [GitHub](https://github.com/bryantky/MAPNet)

---

#### 4.2.6 CTNet-SSL

**Recommendation:** ⭐⭐⭐
**Method type:** CNN-Transformer hybrid network + contrastive self-supervised learning
**Input:** radar signal
**Supervision:** self-supervised / contrastive learning / few-label fine-tuning
**Strictly matches radar modulation recognition:** yes
**Suitable for:** limited labeled data, self-supervised pretraining, contrastive learning

This project targets the lack of labeled training data in radar signal modulation recognition and is useful for studying CNN-Transformer hybrid structures and contrastive self-supervised learning. Since the public documentation is limited, the data, training scripts, and parameter settings should be carefully checked before reproduction.

**Link:** [GitHub](https://github.com/wanan0414/CTNet-SSL)

---

#### 4.2.7 SEMTN

**Recommendation:** ⭐⭐⭐
**Method type:** Self-enhanced Multidimensional Taylor Network
**Input:** `.mat` radar data
**Supervision:** supervised classification
**Strictly matches radar modulation recognition:** yes
**Suitable for:** robust recognition under multiple channels and multiple SNRs

The SEMTN repository contains data files and MATLAB code under different channel conditions such as AWGN, Nakagami, Rayleigh, and Rician. Its README is brief, so users need to parse the data format and training workflow by themselves.

**Link:** [GitHub](https://github.com/Guqih/SEMTN)

---

#### 4.2.8 rsrc-for-pub

**Recommendation:** ⭐⭐⭐⭐
**Method type:** collection of code and datasets from multiple radar signal recognition papers
**Input:** various forms, including time-frequency images, signal sequences, and dataset code
**Supervision:** various
**Strictly matches radar modulation recognition:** some subprojects do
**Suitable for:** low-SNR recognition, multi-component radar signal recognition, traditional + deep method comparison

This repository contains multiple radar signal recognition subprojects, such as SFUnet-DCNN and SQDandFE. Some subprojects strictly belong to LPI radar waveform recognition or radar signal recognition, while others belong to multi-component or extended tasks. Users should judge the task setting of each subdirectory separately.

**Link:** [GitHub](https://github.com/stu-cjlu-sp/rsrc-for-pub)

---

#### 4.2.9 Sig2text

**Recommendation:** ⭐⭐⭐
**Method type:** vision-language model / signal-to-text parsing
**Input:** radar waveform time-frequency representation
**Supervision:** supervised parsing
**Strictly matches radar modulation recognition:** partially; it is an extended task
**Suitable for:** modulation recognition + parameter estimation + semantic signal parsing

Sig2text is not a traditional single-label modulation classification method. Instead, it extends radar signal recognition to textual / symbolic parsing. It is suitable for research focusing on “modulation type + parameters + interpretable description”.

**Link:** [Paper](https://arxiv.org/abs/2503.15213)

---

## 5. Recommended Experimental Setup

If the research goal is **radar modulation type recognition**, the following experimental settings are recommended.

---

### 5.1 Main Benchmarks

| Goal                                                      | Recommended Benchmark     |
| --------------------------------------------------------- | ------------------------- |
| Standard spectrogram / image classification               | AIMC-Spec / AIMC-Spec-v2  |
| 1D I/Q sequence classification                            | DeepRadar2022             |
| Multi-task: modulation recognition + parameter estimation | RadChar                   |
| Few-shot / self-supervised learning                       | RadCharSSL                |
| Joint radar + communication recognition                   | RadarCommDataset          |
| Classic LPI waveform recognition pipeline                 | LPI-Net generated data    |
| Custom intrapulse modulation simulation                   | Liuyh0308 simulation code |

---

### 5.2 Baseline Methods

Recommended baseline methods include:

| Category                    | Methods                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| Traditional features        | Instantaneous frequency + SVM, HOC + SVM, handcrafted TFI features       |
| Time-frequency image models | STFT + CNN, CWD + CNN, WVD + ResNet, CWT + DenseNet                      |
| 1D I/Q models               | 1D-CNN, CNN-LSTM, TCN, Transformer encoder                               |
| Robust recognition          | Denoising + classifier, DNCNet-style model, attention CNN                |
| Few-shot / SSL              | Prototypical network, MAML, contrastive learning, masked signal modeling |
| Open-set recognition        | Metric learning, threshold-based rejection, OpenMax, energy score        |
| Multi-task learning         | Classification + parameter regression                                    |

---

### 5.3 Suggested Pipelines

#### 5.3.1 Standard Supervised Classification Pipeline

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

#### 5.3.2 Low-SNR Robust Recognition Pipeline

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

#### 5.3.3 Few-Shot / Self-Supervised Pipeline

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

## 6. Related Tasks

Radar modulation type recognition is often confused with other tasks. These should be clearly distinguished when using resources.

| Related Task                             | Task Definition                                                    | Relationship to Modulation Recognition                             |
| ---------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Communication AMC                        | Identifies communication signal modulation, such as QPSK, QAM, FSK | Similar methods, but different data and physical mechanisms        |
| Radar pulse deinterleaving               | Assigns mixed PDW pulse streams to different emitters              | A sorting task, not modulation classification                      |
| Radar emitter identification / SEI       | Identifies a specific emitter or device identity                   | Focuses on device fingerprints, not only waveform class            |
| Radar target recognition / SAR ATR       | Identifies target classes, such as aircraft, vehicles, ships       | Target recognition, not signal modulation recognition              |
| Radar spectrum detection                 | Detects whether radar signals exist in wideband spectra            | Detection task; can be a preprocessing step                        |
| Radar activity segmentation              | Performs sample-wise radar activity labeling on long I/Q sequences | Segmentation task, not single-signal classification                |
| Radar jamming recognition                | Identifies jamming styles, such as ISRJ, VGPO, comb jamming        | The object is jamming signals; adjacent to modulation recognition  |
| Multi-component radar signal recognition | Identifies multiple radar signal components in one observation     | A complex extension of modulation recognition                      |
| Signal parameter estimation              | Estimates bandwidth, pulse width, chirp rate, code length, etc.    | Can be combined with modulation recognition in multi-task learning |
| Signal-to-text / symbolic parsing        | Parses radar signals into text or structured parameters            | A higher-level task above modulation recognition                   |

---

## 7. Recommended Reading and Starting Points

### Standard Modulation Recognition Benchmarks

* [AIMC-Spec paper](https://arxiv.org/html/2601.08265v2)
* [AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification)
* [AIMC-Spec-v2 Kaggle](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)
* [DeepRadar2022](https://www.kaggle.com/datasets/khilian/deepradar)

### Classic LPI Radar Waveform Recognition

* [LPI-Radar-Waveform-Recognition](https://github.com/vannguyentoan/LPI-Radar-Waveform-Recognition)
* [MathWorks LPI Radar Waveform Classification Example](https://www.mathworks.com/help/radar/ug/LPI-radar-waveform-classification-using-time-frequency-CNN.html)

### Datasets and Few-Shot / Self-Supervised Learning

* [RadChar](https://github.com/abcxyzi/RadChar)
* [RadCharSSL](https://github.com/abcxyzi/RadCharSSL)
* [RadCharSSL Kaggle](https://www.kaggle.com/datasets/abcxyzi/radcharssl-mlsp-2025)
* [RadarCommDataset](https://github.com/ANDROComputationalSolutions/RadarCommDataset)

### Low-SNR and Robust Recognition

* [DNCNet-pytorch](https://github.com/dumingyang20/DNCNet-pytorch)
* [MAPNet](https://github.com/bryantky/MAPNet)
* [SEMTN](https://github.com/Guqih/SEMTN)
* [rsrc-for-pub](https://github.com/stu-cjlu-sp/rsrc-for-pub)

### Data Generation and Simulation

* [Radar-Intra-Pulse-Modulation-Signal-Simulation](https://github.com/Liuyh0308/Radar-Intra-Pulse-Modulation-Signal-Simulation)
* [Radar-Intra-Pulse-Modulation-Simulation](https://github.com/dumingyang20/Radar-Intra-Pulse-Modulation-Simulation)

### Extended Tasks

* [Sig2text paper](https://arxiv.org/abs/2503.15213)
* [RadDet](https://github.com/abcxyzi/RadDet)
* [RadSeg](https://github.com/abcxyzi/radseg)

---

## 8. Notes

* Compared with communication automatic modulation classification, datasets and open-source code for radar modulation type recognition are much fewer.
* Many papers use self-generated data. Experimental results strongly depend on modulation parameters, SNR range, sampling rate, pulse width, bandwidth, and train-test splits.
* Reports should clearly state whether the input is I/Q sequence, spectrum, STFT, CWD, WVD, CWT, or another representation.
* If synthetic data are used, the signal parameter range, noise model, channel model, SNR settings, and random seed should be disclosed.
* For low-SNR robustness studies, an accuracy-SNR curve should be reported rather than only average accuracy.
* For few-shot studies, the N-way K-shot setting, base / novel class split, number of episodes, and confidence interval should be specified.
* For open-set studies, the known / unknown class split should be specified, and AUROC, OSCR, FPR95 and other metrics should be reported when appropriate.
* If RadarCommDataset or RadioML is used, it should be clearly stated which experiments are radar modulation recognition, communication AMC, or joint radar-communication recognition.
* Radar detection, pulse deinterleaving, specific emitter identification, target recognition, and jamming recognition should not be directly described as radar modulation type recognition.
* The recommendation levels in this repository may be updated as projects, datasets, and reproducibility records change.

---

## 9. Citation and Contribution

This resource list is maintained by **SMARTDSP Lab, School of Informatics, Xiamen University**. It aims to collect public datasets, open-source code, representative methods, and reproducible experimental resources related to **Radar Modulation Type Recognition / Radar Waveform Recognition / Automatic Intrapulse Modulation Classification**.

If this repository is helpful for your study, research, or project development, you are welcome to cite this repository in your paper, report, or project documentation. Please also cite the original papers or project pages of the corresponding methods, datasets, and code repositories.

### Maintenance Notes

* This repository mainly focuses on radar modulation type recognition, LPI radar waveform recognition, automatic intrapulse modulation classification, radar signal recognition, and related tasks.
* During resource organization, we try to annotate task type, data availability, supervision setting, and whether each resource strictly belongs to radar modulation type recognition.
* Since some open-source projects may have incomplete data descriptions, licenses, or reproduction workflows, related judgments may be revised as projects are updated.
* The recommendation level in this repository only reflects the maintainer’s subjective assessment based on task relevance, openness, reproducibility, and documentation completeness.

### How to Contribute

Issues and pull requests are welcome. Contributions may include, but are not limited to:

* newly released radar modulation type recognition datasets;
* open-source implementations related to LPI radar waveform recognition, AIMC, or radar signal recognition;
* reproducible benchmark results;
* corrections about dataset availability, broken links, or license information;
* additional clarification on whether a method strictly belongs to radar modulation type recognition;
* boundary notes on related but different tasks such as communication AMC, radar detection, pulse deinterleaving, and target recognition.

When submitting a new resource, please try to include the following information:

```text
Project name:
Project link:
Task type:
Method type:
Input representation:
Open-source code:
Open-source data:
Labels included:
Strictly radar modulation type recognition:
Reason for recommendation:
Notes:
```
