# Radar Emitter Model Identification Resource Summary

<p align="center">
  <a href="./README.md">简体中文</a> |
  <b>English</b>
</p>

This repository is maintained by the **SmartDSP Lab, School of Informatics, Xiamen University**. It summarizes public resources, representative papers, and reproducible experiment suggestions related to **radar emitter model identification / radar emitter type identification / radar emitter classification**.

The core task of this page is **radar emitter model identification**, namely determining whether an intercepted radar signal, a Pulse Description Word (PDW) parameter sequence, intra-pulse features, or an operational behavior sequence belongs to a **known radar model, equipment family, functional category, or emitter class**.

This resource list follows these principles:

* Prioritize papers and resources directly related to **radar emitter identification, radar emitter model identification, and radar emitter type identification**;
* Datasets labeled only by modulation category, waveform category, or individual device ID are included only as related tasks or auxiliary resources;
* Projects without clear papers, data descriptions, or reproducible value are not recommended in the main text;
* Methods without public data or official code but highly relevant to the task are included under “Representative Papers” rather than “Open-source Resources”.

---

## Quick Takeaways

* **Public datasets for strict radar model identification are extremely scarce.** At present, there is still a lack of stable, public, and reproducible benchmark datasets with real radar model-level labels. Most papers use self-built simulated PDW datasets, simulated radar parameter libraries, internal intercepted data, or self-built model libraries.
* **The public resources closest to model identification** are datasets such as RadarCommDataset, which contain radar functional categories or signal-type labels. However, they are still not strict radar model-level datasets.
* **Representative papers on model identification** should focus on PDW parameter-set identification, radar emitter identification, open-set radar emitter identification, transfer learning / online learning, and agile-waveform emitter identification.
* **Experimental reports must clearly explain the meaning of labels.** Whether the labels represent radar models, functional categories, waveform categories, modulation categories, individual device IDs, or working modes determines whether the task is strict model identification.

---

## Table of Contents

* [Quick Takeaways](#quick-takeaways)
* [1. Task Overview](#1-task-overview)

  * [1.1 What Is Radar Emitter Model Identification?](#11-what-is-radar-emitter-model-identification)
  * [1.2 Differences from Related Tasks](#12-differences-from-related-tasks)
  * [1.3 Common Input Features](#13-common-input-features)
  * [1.4 Common Evaluation Metrics](#14-common-evaluation-metrics)
* [2. Method Taxonomy and Task Boundaries](#2-method-taxonomy-and-task-boundaries)

  * [2.1 Methods for Strict Model Identification](#21-methods-for-strict-model-identification)
  * [2.2 Related but Non-equivalent Methods](#22-related-but-non-equivalent-methods)
  * [2.3 Relationship Between Methods and Task Definitions](#23-relationship-between-methods-and-task-definitions)
* [3. Datasets and Public Resources](#3-datasets-and-public-resources)

  * [3.1 Dataset Overview](#31-dataset-overview)
  * [3.2 Notes on Model Identification Data](#32-notes-on-model-identification-data)

    * [3.2.1 Real Radar Model-level Datasets](#321-real-radar-model-level-datasets)
    * [3.2.2 RadarCommDataset](#322-radarcommdataset)
    * [3.2.3 Simulated Radar Model Libraries](#323-simulated-radar-model-libraries)
* [4. Representative Methods and Papers](#4-representative-methods-and-papers)

  * [4.1 Paper Overview](#41-paper-overview)
  * [4.2 Brief Introduction to Methods](#42-brief-introduction-to-methods)
  * [4.3 Detailed Notes on Papers and Methods](#43-detailed-notes-on-papers-and-methods)
* [5. Recommended Experimental Settings](#5-recommended-experimental-settings)

  * [5.1 Suggested Baselines](#51-suggested-baselines)
  * [5.2 Suggested Pipelines](#52-suggested-pipelines)
* [6. Notes](#6-notes)
* [7. Related Contributions and Patent Layout from Our Lab](#7-related-contributions-and-patent-layout-from-our-lab)
* [8. Citation and Contribution](#8-citation-and-contribution)

---

## 1. Task Overview

### 1.1 What Is Radar Emitter Model Identification?

Radar emitter model identification refers to determining the **radar model, equipment family, functional category, operating regime, known emitter class, or unknown class** of a target emitter based on radar radiation signals intercepted by electronic reconnaissance, radar warning, signal intelligence, or electromagnetic spectrum monitoring systems. It is not a single fixed classification task; instead, it can be further divided into multiple identification problems according to input data type, sample organization, and label granularity.

From the perspective of input data, radar emitter model identification can be based on **PDW parameter data**, as well as **intermediate-frequency signals, baseband IQ signals, intra-pulse sampled data, time-frequency images, spectrograms, or multimodal fused features**. PDW data usually comes from parameter measurements of pulse signals by electronic reconnaissance receivers, including time of arrival, pulse width, carrier frequency, amplitude, direction of arrival, and other information. IQ or intermediate-frequency data preserves more complete intra-pulse modulation, frequency variation, phase variation, and waveform-structure information.

From the perspective of sample granularity, model identification can be performed at the **single-pulse level** or the **multi-pulse sequence level**. Single-pulse identification takes one pulse or a short segment of intra-pulse sampled signal as input and outputs the possible emitter category of that pulse. Multi-pulse sequence identification takes a continuous pulse sequence, an emitter trajectory, or an intercepted time slice as input, and uses sequence information such as PRI variation, frequency agility, pulse-width variation, amplitude variation, direction-of-arrival variation, and operational behavior patterns to determine the radar emitter category corresponding to that sequence. For agile radars, multifunction radars, and modern cognitive radars, sequence-level modeling usually better captures radar behavioral characteristics than single-pulse identification.

From the perspective of output labels, identification results may also have different levels of granularity. Coarse-grained labels can be **radar functional categories**, such as search radar, tracking radar, height-finding radar, airborne detection radar, and ground surveillance radar. Medium-grained labels can be **radar regimes or equipment families**, such as a certain type of pulse-Doppler radar, phased-array radar, frequency-modulated continuous-wave radar, or a specific equipment family. Fine-grained labels can be **specific radar models or known emitter classes**. In open-set scenarios, the model also needs to determine whether the input belongs to an unknown radar emitter that does not appear in the training set.

A typical radar emitter model identification pipeline can be represented as:

```text
Intercepted radar signal / PDW data / IF signal / IQ signal
        ↓
Preprocessing, pulse detection, parameter measurement, or time-frequency transformation
        ↓
Single-pulse feature extraction / sequence modeling / intra-pulse representation learning / multimodal fusion
        ↓
Classifier, metric-learning model, or open-set recognizer
        ↓
Radar model / equipment family / functional category / known emitter class / unknown class
```

If the input is single-pulse PDW data, one pulse can usually be represented as:

```text
pi = [TOA, RF/CF, PW, PA, DOA/AOA, ...]
```

The model then learns a mapping from a single-pulse parameter vector to a class label:

```text
f(pi) -> yi
```

where `yi` can indicate that the pulse is assigned to a known radar category, an emitter category, or an intermediate result for subsequent sequence identification.

If the input is a multi-pulse PDW sequence, one sample can be represented as:

```text
P = [p1, p2, ..., pT]
```

where each pulse `pt` may contain:

```text
pt = [TOA, PRI, RF/CF, PW, PA, DOA/AOA, ...]
```

The model then learns a mapping from a pulse sequence to an emitter category:

```text
f(P) -> y
```

where `y` usually denotes the radar model, radar type, or known emitter class corresponding to the sequence, emitter trajectory, or time slice.

If the input is an intermediate-frequency signal or baseband IQ signal, one sample can be represented as:

```text
x = [I1 + jQ1, I2 + jQ2, ..., IN + jQN]
```

or as an intermediate-frequency sampled sequence:

```text
s = [s1, s2, ..., sN]
```

The model can then learn intra-pulse modulation, frequency variation, phase variation, envelope structure, and spectral features directly from the raw waveform, and perform emitter identification:

```text
f(x) -> y
```

In more complex scenarios, a model can also use both PDW parameters and IQ / intermediate-frequency data for multimodal fusion:

```text
f(P, x) -> y
```

Here, the PDW sequence provides inter-pulse behavioral features, while IQ or intermediate-frequency signals provide intra-pulse waveform details. The former is more suitable for characterizing radar operational patterns and parameter agility, while the latter is more suitable for characterizing intra-pulse modulation, hardware features, and waveform details. In practical radar emitter model identification systems, both types of information often need to be combined to achieve stable identification performance under complex electromagnetic environments, low signal-to-noise ratios, multi-emitter interleaving, and the presence of unknown targets.

Therefore, radar emitter model identification can be understood as a multi-granularity, multimodal, and multi-level identification task. It can be single-pulse-level classification or pulse-sequence-level classification; it can be based on PDW parameters, intermediate-frequency signals, IQ signals, spectrograms, or time-frequency images; and its output can be the category of a single pulse, a pulse sequence, an emitter trajectory, a time slice, or multiple radar emitter categories present in a scene.

### 1.2 Differences from Related Tasks

| Task                                                | Output                                                                                            | Relationship to Model Identification                                     | Equivalent to Model Identification? |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| Radar pulse deinterleaving                          | Emitter cluster to which each pulse belongs                                                       | Preprocessing or parallel task before model identification               | No                                  |
| Radar waveform / intra-pulse modulation recognition | Waveform or modulation categories such as LFM, Barker code, Frank code, P1-P4, Costas, etc.       | Can provide important clues for model identification                     | No                                  |
| Radar functional-category recognition               | Functional categories such as search, tracking, height-finding, imaging, airborne detection, etc. | Can be viewed as coarse-grained model identification                     | Partially                           |
| Radar emitter model identification                  | Radar model, equipment family, known emitter class                                                | Core task of this page                                                   | Yes                                 |
| Specific emitter identification                     | A specific device or individual transmitter ID                                                    | Finer-grained than model identification                                  | No                                  |
| RF fingerprint identification                       | Hardware identity of a transmitter                                                                | Can provide methodological reference for individual radar identification | No                                  |
| Radar working-mode recognition                      | Modes such as search, tracking, guidance, height-finding, etc.                                    | Can support model identification and intent inference                    | No                                  |
| Radar spectrum detection                            | Whether radar signals exist in the spectrum and their frequency-domain locations                  | A detection task, not model classification                               | No                                  |
| Radar jamming recognition                           | Jamming pattern or jamming category                                                               | Electronic-countermeasure-related task                                   | No                                  |

---

### 1.3 Common Input Features

| Feature or Representation       | Meaning                                                                                                  | Use                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Raw IQ                          | Complex baseband sampled signal                                                                          | End-to-end 1D CNNs, Transformers, self-supervised pretraining                     |
| Time-frequency image            | STFT, CWT, Wigner-Ville distribution, pseudo Wigner-Ville distribution, Choi-Williams distribution, etc. | Recognition of intra-pulse structure, modulation style, and waveform regime       |
| TOA                             | Time of arrival                                                                                          | PRI estimation and mode-sequence analysis                                         |
| PRI / DTOA                      | Pulse repetition interval / difference in time of arrival                                                | Distinguishing repetition periods and PRI modulation patterns                     |
| RF / CF                         | Radio frequency / carrier frequency                                                                      | Identifying carrier frequency, frequency hopping, and frequency agility           |
| PW                              | Pulse width                                                                                              | Distinguishing pulse parameters and transmission regimes                          |
| PA / AMP                        | Pulse amplitude                                                                                          | Auxiliary identification, but affected by propagation paths and receivers         |
| DOA / AOA                       | Direction of arrival / angle of arrival                                                                  | Spatial auxiliary features in multi-emitter scenarios                             |
| Intra-pulse modulation features | LFM, NLFM, BPSK, Barker code, Frank code, P1-P4, etc.                                                    | Can support radar-regime inference, but is not equivalent to model identification |
| Working-mode sequence           | Mode-transition sequence such as search, tracking, guidance, etc.                                        | Working-mode recognition, model-aided identification, and intent inference        |
| Hardware fingerprint features   | Transmitter non-ideal characteristics                                                                    | Specific emitter identification                                                   |

---

### 1.4 Common Evaluation Metrics

| Metric                     | Description                                                      | Recommended Scenario                                      |
| -------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| Accuracy                   | Classification accuracy                                          | Balanced closed-set identification                        |
| Balanced accuracy          | Average recall across classes                                    | Imbalanced datasets                                       |
| Macro F1                   | Macro-average F1 across classes                                  | Multi-class imbalanced identification                     |
| Weighted F1                | F1 weighted by sample count                                      | When class frequencies differ greatly                     |
| Confusion matrix           | Shows confusion between classes                                  | Analyzing easily confused models or functional categories |
| Top-k accuracy             | Whether the correct class appears among the top-k predictions    | Large model libraries                                     |
| AUROC                      | Ability to distinguish known and unknown classes                 | Open-set / out-of-distribution identification             |
| AUPR                       | Detection ability under imbalanced positive and negative samples | Open-set / out-of-distribution identification             |
| FPR95                      | False positive rate when true positive rate is 95%               | Open-set identification                                   |
| Expected calibration error | Confidence calibration error                                     | Reliability evaluation                                    |
| Cross-domain accuracy      | Accuracy across SNRs, receivers, or scenarios                    | Generalization evaluation                                 |
| Few-shot accuracy          | Accuracy under N-way K-shot settings                             | Few-shot model identification                             |

---

## 2. Method Taxonomy and Task Boundaries

### 2.1 Methods for Strict Model Identification

The following methods directly target radar model, radar type, functional category, or known emitter-class identification.

| Method Type                                          | Task Match | Description                                                                                                    |
| ---------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| Parameter-library matching / threat-library matching | High       | Matches RF, PRI, PW, scan period, working mode, and other parameters against known radar libraries             |
| Expert systems / rule-based reasoning                | High       | Identifies radar types or functional categories based on electronic reconnaissance expert rules                |
| PDW parameter-set classification                     | High       | Uses PDW features such as RF, PRI, PW, PA, and DOA for emitter-class identification                            |
| Traditional machine-learning classification          | High       | SVM, random forest, gradient boosting trees, KNN, etc. for parameter features or statistical features          |
| Working-mode sequence modeling                       | High       | HMM, LSTM, Transformer, etc. for mode sequences and multi-pulse behavior identification                        |
| Transfer learning / online learning                  | High       | Addresses new scenarios, few-shot samples, and dynamic updates of radar model libraries                        |
| Open-set model identification                        | High       | Identifies unknown emitters or unknown models during testing                                                   |
| Multi-domain feature fusion                          | High       | Fuses PDW, intra-pulse modulation, time-frequency features, and working-mode features for model identification |

---

### 2.2 Related but Non-equivalent Methods

The following methods are highly related to model identification, but should not be directly described as strict model identification.

| Method Type                                                      | Relationship to Model Identification   | Description                                                                                                                  |
| ---------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Radar waveform / modulation recognition                          | Auxiliary task                         | Waveform categories can serve as model clues, but waveform category alone is not a radar model                               |
| Radar functional-category recognition                            | Coarse-grained related task            | If labels are radar functional categories, the task can be viewed as type recognition, but not specific model identification |
| Specific emitter identification / RF fingerprint identification  | Finer-grained related task             | Identifies a specific device individual, usually finer-grained than model identification                                     |
| Radar spectrum detection                                         | Preceding detection task               | Determines whether radar signals exist in the spectrum, without outputting model labels                                      |
| Radar jamming recognition                                        | Related electronic-countermeasure task | Identifies jamming patterns rather than radar emitter models                                                                 |
| CNNs / Vision Transformers on modulation-classification datasets | Auxiliary methods                      | Can be used for pretraining or waveform recognition, but are not the main line of model identification                       |

---

### 2.3 Relationship Between Methods and Task Definitions

| Method Family                                                   | Fits Model Identification? | Label Requirement                                                  | Description                                                      |
| --------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Parameter-library / threat-library matching                     | Yes                        | Model, functional-category, or equipment-family labels             | Classic engineering approach, but depends on a complete database |
| PDW parameter-set classification                                | Yes                        | Emitter-class labels                                               | Closest to emitter identification in electronic reconnaissance   |
| Traditional machine learning + PDW features                     | Yes                        | Emitter-class labels                                               | Recommended as basic baselines                                   |
| Deep networks + PDW / IQ / time-frequency features              | Yes                        | Emitter-class labels                                               | Suitable for complex nonlinear feature modeling                  |
| Transfer learning / online learning                             | Yes                        | Source-domain or partially labeled target-domain classes           | Suitable for scenario changes and model-library updates          |
| Open-set identification                                         | Yes                        | Known-class labels, with unknown classes for testing or validation | Closer to realistic electronic reconnaissance scenarios          |
| Waveform / modulation recognition                               | Partially related          | Waveform or modulation labels                                      | Cannot be directly equated with model identification             |
| Specific emitter identification / RF fingerprint identification | Partially related          | Individual device IDs                                              | Cannot be directly equated with model identification             |
| Spectrum detection / jamming recognition                        | Related task               | Detection labels or jamming labels                                 | Not model identification                                         |

---

## 3. Datasets and Public Resources

### 3.1 Dataset Overview

| Dataset / Resource              | Label Meaning                                                                       | Strict Model Identification? | Recommendation | Link                                                                      | Notes                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------- | -------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Real radar model-level datasets | Radar model / equipment family / known emitter class                                | Yes                          | ⭐              | No stable public benchmark available                                      | Real data is sensitive, and public reproducible resources are extremely scarce                                  |
| RadarCommDataset                | Radar / communication signal categories, including some radar functional categories | Partially                    | ⭐⭐⭐            | [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset) | Suitable as a starting point for functional-category recognition or radar / communication signal classification |

---

### 3.2 Notes on Model Identification Data

#### 3.2.1 Real Radar Model-level Datasets

Strict radar emitter model identification requires labels corresponding to real radar models, equipment families, functional categories, or known emitter classes. Because real radar signals usually involve sensitive information, publicly downloadable, reproducible datasets with clear model-level labels are extremely rare.

Therefore, this repository currently does not list any real radar model-level public dataset as a primary benchmark. If the research objective is strict model identification, it is recommended to use a self-built simulated model library or compliant internal data, while clearly explaining the label design and simulation parameters.

#### 3.2.2 RadarCommDataset

**Resource Name:** RadarCommDataset: Radar & Communication Signal and Modulation Classification Dataset
**Paper:** Dataset for Modulation Classification and Signal Type Classification for Multi-task and Single Task Learning
**Publication:** Computer Networks, 2021, Volume 199, Article 108441
**Resource Link:** [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset) / [Paper](https://doi.org/10.1016/j.comnet.2021.108441)
**Resource Type:** Public wireless-signal dataset / radar and communication signal classification dataset
**Task Type:** Signal-type classification, modulation classification, multi-task learning
**Strict Model Identification?:** No. The labels are not specific radar models or equipment families.
**Recommended Use:** Radar functional-category recognition, radar / communication signal discrimination, cross-SNR robustness experiments, and public baseline construction

**Summary:**

RadarCommDataset is a public dataset for intelligent wireless-signal classification. It contains both radar waveforms and communication waveforms, and provides modulation labels, signal-type labels, SNR labels, and IQ sampled data. The paper focuses on modulation classification and signal-type classification, supporting both single-task learning and multi-task learning. For radar emitter model identification, the value of this dataset lies in the fact that it contains several radar functional signal classes, such as airborne detection radar, airborne range-measurement radar, air-to-ground moving-target indication radar, terrain-mapping radar, and radar altimeter signals. Therefore, it can serve as a public experimental entry point for “radar functional-category recognition” or “radar / communication signal classification”.

However, it should be noted that the label granularity of RadarCommDataset is signal type and functional category, rather than specific radar model, equipment family, or individual emitter. Therefore, if this dataset is cited in a model-identification README, it must be clearly stated that it is only a related substitute task and should not be described as a strict real radar model-identification benchmark.

**Data Contents:**

* Data format: HDF5;
* Fields include `modulation`, `signal`, `snr`, and `sample`;
* Each sample is a 256 × 1 tensor consisting of 128 I-component points and 128 Q-component points;
* Sampling rate: 10 MS/s;
* Each waveform contains 700 snapshot samples;
* SNR ranges from -20 dB to 18 dB, with a step size of 2 dB;
* Data versions include a dynamic-channel version, an AWGN-only version, and a 2.45 GHz over-the-air acquisition version.

**Highlights:**

* Provides both radar and communication signals, suitable for radar / communication signal discrimination;
* The same sample has both signal-category and modulation-category labels, suitable for multi-task learning;
* Covers a wide SNR range, suitable for low-SNR robustness experiments;
* Provides data-reading and visualization scripts, making it convenient to build baseline models;
* Can be used as public auxiliary data for model-identification research, but cannot replace real model-level data.

**Usage Boundaries:**

* It can be described as a “public dataset related to radar functional-category recognition”;
* It can be used for pretraining, auxiliary validation, or public substitute experiments;
* It is not recommended to describe it as a “standard dataset for radar model identification”;
* Modulation categories or signal functional categories should not be directly equated with radar models;
* If used in paper experiments, the label granularity and task boundary should be clearly specified.

#### 3.2.3 Simulated Radar Model Libraries

When public real model-level data is unavailable, a reasonable approach is to construct a simulated radar model library. Each “model” should not be defined by a single waveform category alone, but by multiple parameters, for example:

```text
Radar model = {
  RF / CF distribution,
  PRI type and parameters,
  PW range,
  intra-pulse modulation type,
  working-mode sequence,
  scanning mode,
  parameter agility pattern
}
```

This approach is closer to model identification than simple modulation classification. Experiments should clearly specify the parameter range of each model, inter-class differences, intra-class variation, train-test split, and whether distribution shift exists.

---

## 4. Representative Methods and Papers

This section only includes papers directly related to **radar emitter identification / radar model identification / radar type identification / open-set radar emitter identification**. Papers on modulation recognition, specific emitter identification, spectrum detection, and jamming recognition are not included in the main table of this section.

### 4.1 Paper Overview

| Method / Direction                             | Representative Paper                                                                                                         | Link                                                                                                                                                                       | Code / Project                                           | Task Match     | Description                                                                                                                                                                                                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Parameter-set clustering and classification    | Radar Emitter Recognition Based on Parameter Set Clustering and Classification                                               | [Paper](https://www.mdpi.com/2072-4292/14/18/4468) / [DOI](https://doi.org/10.3390/rs14184468)                                                                             | Paper method                                             | High           | Uses PDW parameter-set clustering, frequent-item construction, and decision-tree classification for radar emitter identification. The task is close to model / type identification and is suitable as a traditional machine-learning baseline.   |
| Agile-waveform emitter identification          | Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism                | [PDF](https://www.radioeng.cz/fulltexts/2021/21_04_0704_0712.pdf)                                                                                                          | [GitHub](https://github.com/fengyuntian2009/PDWSequence) | High           | Identifies agile-waveform radar emitter types based on deinterleaved PDW sequences, using dynamic CNNs, LSTMs, and attention mechanisms. Currently one of the closest directions with an open-source project.                                    |
| Transfer learning and online learning          | Radar Emitter Identification under Transfer Learning and Online Learning                                                     | [Paper](https://www.mdpi.com/2078-2489/11/1/15) / [DOI](https://doi.org/10.3390/info11010015)                                                                              | Paper method                                             | High           | Focuses on radar emitter identification under few target-domain samples, environmental changes, and dynamic updating, using transfer learning, TrAdaBoost-SVM, expectation maximization, and online learning.                                    |
| Attribute-specific recurrent sequence modeling | Radar Emitter Classification with Attribute-specific Recurrent Neural Networks                                               | [arXiv](https://arxiv.org/abs/1911.07683)                                                                                                                                  | Paper method                                             | High           | Models different attributes in PDW pulse streams separately and then fuses temporal features. Suitable for emitter type identification using multi-attribute sequences such as RF, PW, PRI, AOA, and PA.                                         |
| Attention-based multi-RNNs                     | Radar Emitter Classification With Attention-Based Multi-RNNs                                                                 | [DOI](https://doi.org/10.1109/LCOMM.2020.2995842) / [PDF](https://www.researchgate.net/publication/341533342_Radar_Emitter_Classification_With_Attention-Based_Multi-RNNs) | Paper method                                             | High           | Uses multiple RNN branches to process different pulse-stream features and fuses them with attention. Suitable for radar emitter classification under complex pulse-stream conditions.                                                            |
| LSTM-based radar emitter type identification   | Identification of Radar Emitter Type with Recurrent Neural Networks                                                          | [PDF](https://sspd.eng.ed.ac.uk/sites/sspd.eng.ed.ac.uk/files/attachments/basicpage/20200916/S2%20Apfeld.pdf) / [DOI](https://doi.org/10.1109/SSPD47486.2020.9271988)      | Paper method                                             | High           | Uses LSTM to model multifunction radar emission-behavior sequences, focusing on radar resource management and behavioral grammar differences. Suitable for agile / multifunction radar type identification.                                      |
| 1D convolutional attention recognition         | Radar Emitter Signal Recognition Based on One-Dimensional Convolutional Neural Network with Attention Mechanism              | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7664421/) / [MDPI](https://www.mdpi.com/1424-8220/20/21/6350)                                                             | Paper method                                             | Medium to High | Learns identification features directly from one-dimensional radar signal sequences and uses attention to highlight key features. More oriented toward intra-pulse signal recognition, but can serve as a radar emitter identification baseline. |
| Knowledge-graph-driven CNN                     | A Knowledge Graph-Driven CNN for Radar Emitter Identification                                                                | [Paper](https://www.mdpi.com/2072-4292/15/13/3289) / [DOI](https://doi.org/10.3390/rs15133289)                                                                             | Paper method                                             | High           | Introduces knowledge graphs into radar emitter identification, using expert knowledge and attribute relationships to assist CNN feature selection and model design. Suitable for knowledge-enhanced identification frameworks.                   |
| Joint multi-radar sorting and recognition      | A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and Time-Frequency Convolution Network | [Paper](https://www.sciencedirect.com/science/article/pii/S1051200425000272)                                                                                               | Paper method                                             | Medium to High | Handles both sorting and recognition in interleaved pulse streams. It first performs multi-radar sorting using hierarchical clustering, then identifies emitters with a time-frequency convolutional network.                                    |

---

### 4.2 Brief Introduction to Methods

#### 4.2.1 Parameter-set Clustering and Classification

“Radar Emitter Recognition Based on Parameter Set Clustering and Classification” is a representative traditional machine-learning method closely related to radar emitter model identification. The method mainly uses PDW parameters and radar pulse parameter sets as inputs. It first clusters parameter sets, extracts stable parameter combinations and frequent items, and then feeds these structured features into a classifier to identify radar emitter categories. Unlike methods that directly recognize modulation types, this method focuses on the overall parameter-level behavioral characteristics of radar emitters, making it more suitable for PDW / parameter-set-based radar model or emitter type identification.

#### 4.2.2 Agile-waveform Emitter Identification

“Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism” addresses the identification of radar emitters with rapidly changing parameters. The paper defines the input as deinterleaved PDW pulse sequences, where each pulse contains features such as PA, CF, PW, PRI, and AOA. It combines dynamic convolutional networks, LSTMs, and attention mechanisms for joint modeling. The convolutional network extracts local parameter-combination features, the LSTM learns temporal variation patterns in the pulse sequence, and the attention mechanism enhances the contribution of key pulses and key features. This method is currently a representative approach close to “multi-pulse PDW sequence radar emitter type identification” and has a corresponding open-source project, making it a priority reproduction target.

#### 4.2.3 Transfer Learning and Online Learning

“Radar Emitter Identification under Transfer Learning and Online Learning” proposes an identification framework based on transfer learning and online learning to address target-domain sample scarcity, signal distribution variation, and dynamic model updating in radar emitter identification. The core value of this direction is to handle domain shift, which is common in real electronic reconnaissance environments. Training data and actual intercepted data may come from different environments, SNR conditions, or operating conditions. The paper uses existing source-domain samples to support target-domain identification and gradually updates the model through an online-learning mechanism, enabling adaptation to new environments and new samples. This direction is highly suitable for supporting radar emitter model identification research in practical deployment scenarios.

#### 4.2.4 Attribute-specific Recurrent Neural Network Sequence Modeling

“Radar Emitter Classification with Attribute-specific Recurrent Neural Networks” is a radar emitter classification method based on multi-attribute PDW sequences. The method argues that different PDW attributes have different statistical properties and temporal variation patterns. Therefore, RF, PW, PRI, AOA, PA, and other features should not simply be concatenated and fed into a single model. Instead, independent recurrent neural network branches should be designed for different attributes. Through attribute-specific recurrent neural networks, the model can separately learn the temporal patterns of different pulse parameters and then fuse the sequence features to complete emitter classification. This method is suitable for building deep-learning baselines based on PDW sequences.

#### 4.2.5 Attention-based Multi-RNNs

“Radar Emitter Classification With Attention-Based Multi-RNNs” further develops the idea of multi-attribute sequence modeling. It uses multiple recurrent neural network branches to process different pulse-stream features and applies an attention mechanism to adaptively weight the importance of different feature branches. This method is suitable for complex electromagnetic environments, because the discriminative contribution of different parameters varies under different noise conditions, working modes, or radar types. The attention mechanism helps the model emphasize more discriminative feature dimensions and temporal segments. Therefore, this direction is suitable as a representative deep-fusion model for PDW-sequence radar emitter classification.

#### 4.2.6 LSTM-based Radar Emitter Type Identification

“Identification of Radar Emitter Type with Recurrent Neural Networks” focuses on multifunction radar emitter type identification using LSTMs to model radar emission-behavior sequences. The emphasis is not on static classification of individual pulse parameters, but on using the implicit “behavioral grammar” or resource-management patterns in pulse sequences to distinguish different radar types. For agile radars, multifunction radars, and cognitive radars, different radars may have similar parameter ranges, but their workflows, task-switching rules, and emission-behavior sequences may differ. LSTMs are suitable for capturing such long-term temporal dependencies, making this method appropriate for radar emitter type identification and behavior-pattern-aided identification.

#### 4.2.7 1D Convolutional Attention Recognition

“Radar Emitter Signal Recognition Based on One-Dimensional Convolutional Neural Network with Attention Mechanism” uses a one-dimensional convolutional neural network and attention mechanism to extract discriminative features directly from radar signal sequences. Unlike PDW-parameter-based identification methods, this type of method learns local structural features from raw one-dimensional signals or intra-pulse sampled sequences, reducing dependence on manually designed features. The attention mechanism strengthens key segments and features in the signal sequence, making the model focus more on components that contribute to emitter discrimination. This method is closer to “radar emitter signal recognition” than pure PDW-based model identification, but it can serve as an intra-pulse signal branch or end-to-end identification baseline.

#### 4.2.8 Knowledge-graph-driven CNN

“A Knowledge Graph-Driven CNN for Radar Emitter Identification” combines radar domain knowledge with deep-learning models using a knowledge graph to assist radar emitter identification. The core idea is to organize expert experience, parameter relationships, emitter attributes, and category associations into a knowledge graph, and then use this prior information to guide feature selection, model-structure design, or identification decisions. Compared with fully end-to-end CNN methods, knowledge-graph-driven models emphasize interpretability and the use of prior knowledge. This is closer to practical electronic reconnaissance workflows that rely on expert rules, parameter libraries, and emitter knowledge bases. This direction is suitable as a representative of “knowledge-enhanced radar model identification” or “rule-and-deep-learning hybrid identification”.

#### 4.2.9 Joint Multi-radar Sorting and Recognition

“A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and Time-Frequency Convolution Network” jointly considers sorting and recognition in interleaved multi-radar pulse streams. The method first uses hierarchical clustering to sort interleaved PDW or pulse streams, separating pulses from different emitters into different sequences as much as possible. Then, it performs time-frequency analysis and uses a convolutional network to identify the sorted signals. This method is closer to a real electronic reconnaissance processing pipeline, because actual received data is often not a clean pulse sequence from a single radar, but a mixed scenario involving multiple simultaneous emitters, pulse overlap, and complex parameter variations. Therefore, this direction is suitable as a representative of “joint sorting-recognition modeling” or “radar model identification in complex multi-emitter scenarios”.

---

### 4.3 Detailed Notes on Papers and Methods

This section expands only on the papers already listed in the table above, without adding new papers. The notes include summaries, method categories, publication venues, input data, label granularity, whether the task is strict model identification, code availability, highlights, and usage boundaries.

#### 4.3.1 Radar Emitter Recognition Based on Parameter Set Clustering and Classification

-**Chinese Title:** 基于参数集聚类与分类的雷达辐射源识别
-**Publication:** Remote Sensing, 2022, 14(18):4468
**Paper Link:** [Paper](https://www.mdpi.com/2072-4292/14/18/4468) / [DOI](https://doi.org/10.3390/rs14184468)
**Method Category:** PDW parameter-set modeling / traditional machine learning / clustering feature extraction + decision-tree classification
**Input Data:** Radar pulse trains or PDW parameters, mainly focusing on PRI / repetition-frequency features, while other pulse parameters can also be embedded
**Label Granularity:** Radar emitter category / radar type
**Strict Model Identification?:** Basically yes. It is a parameter-level radar emitter identification method.
**Code Availability:** No official open-source code found

**Summary:**

This paper addresses measurement noise, missing pulses, and false pulses in intercepted radar pulses, and proposes a radar emitter identification method based on parameter-set clustering and decision-tree classification. Traditional methods often directly compute statistical features such as the mean and variance of pulse parameters. However, when pulse parameters are affected by noise or when radars have multiple repetition-frequency modes, simple statistical features can become distorted. This method first uses Mean Shift clustering to extract frequent items and clustering centers related to repetition frequency from pulse trains. It then constructs variable-length feature vectors based on the number of frequent items and their center values, and finally uses an improved multi-way decision tree to identify unknown radar pulse trains.

The significance of this method is that it does not treat each individual pulse as an independent classification sample. Instead, it extracts stable repetition-frequency structures from a pulse train or parameter set, which is closer to the idea of “parameter-set identification” in traditional electronic reconnaissance. It is suitable as a traditional machine-learning baseline based on PDW parameters, especially for comparison with LSTM, SVM, or deep sequence models.

**Highlights:**

* Uses Mean Shift clustering to extract frequent parameter items from pulse trains;
* Constructs radar parameter-set features using clustering centers and data-count information;
* Uses an improved decision tree to handle inconsistent feature dimensions across samples;
* More robust than simple statistical features under noise and changes in data volume;
* The task is close to radar emitter type identification and is suitable for the “PDW parameter-set classification” direction.

**Applicable Scenarios:**

* Closed-set radar emitter identification based on PDW parameters;
* Radar signals with fixed PRI, staggered PRI, or obvious repetition-frequency patterns;
* Experiments requiring interpretable traditional machine-learning baselines;
* Model-identification tasks that need to emphasize “parameter-set features” rather than “single-pulse modulation categories”.

**Limitations:**

* Mainly relies on pulse parameters and repetition-frequency structures, with limited use of intra-pulse details;
* The data is simulated or from controlled scenarios, and still differs from real complex electromagnetic environments;
* Highly agile radars, rapidly switching modes, or unstable repetition-frequency structures may require extensions;
* No official open-source code is provided, so the clustering and decision-tree pipeline must be implemented independently for reproduction.

---

#### 4.3.2 Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism

**Chinese Title:** 基于混合深度神经网络与注意力机制的敏捷波形雷达辐射源识别
**Publication:** Radioengineering, 2021, 30(4):704–712
**Paper Link:** [PDF](https://www.radioeng.cz/fulltexts/2021/21_04_0704_0712.pdf)
**Code Link:** [GitHub: PDWSequence](https://github.com/fengyuntian2009/PDWSequence)
**Method Category:** PDW sequence modeling / dynamic CNN + LSTM + attention mechanism / supervised deep classification
**Input Data:** Deinterleaved PDW pulse sequences, including PA, CF, PW, PRI, AOA, and other parameters
**Label Granularity:** Agile-waveform radar emitter category
**Strict Model Identification?:** Relatively yes. It belongs to multi-pulse PDW sequence-level radar emitter identification.
**Code Availability:** Related GitHub project available

**Summary:**

This paper focuses on the identification of agile-waveform radar emitters. For agile-waveform radars, carrier frequency, pulse width, repetition frequency, and other parameters can vary rapidly, making it difficult to accurately represent the target emitter using only one fixed group of PDW parameters. The paper defines the input as deinterleaved pulse sequences, where each pulse consists of amplitude, carrier frequency, pulse width, PRI, direction of arrival, and other parameters. The method first performs distributed representation on pulse data to generate high-dimensional sparse features. It then uses a dynamic convolutional neural network to extract structural detail features at different levels, uses an LSTM to model temporal variation patterns in the pulse sequence, and finally uses an attention mechanism to fuse structural and temporal features before Softmax classification.

This method is suitable as a representative of “PDW-sequence-based model identification”, because it explicitly focuses on radar emitter categories rather than simple modulation classification. It also has a corresponding open-source project, making it one of the more reproducible entries in the current model-identification resource list.

**Highlights:**

* Targets agile-waveform radars rather than ordinary fixed-parameter radars;
* Uses deinterleaved multi-pulse PDW sequences as input;
* Dynamic CNN extracts local structural features, while LSTM extracts temporal features;
* The attention mechanism fuses structural and temporal information and reduces noise influence;
* A public GitHub project is available, making it suitable as a priority reproduction target.

**Applicable Scenarios:**

* Agile-waveform radar emitter type identification;
* Multi-pulse PDW sequence classification;
* Tasks requiring joint modeling of local parameter combinations and long-term temporal variation patterns;
* Deep-learning baseline for PDW sequence identification.

**Limitations:**

* The input depends on “deinterleaved pulse sequences”, so a preceding deinterleaving step is usually required;
* The training stage depends on labels and is not an unsupervised method;
* Whether the data and labels cover real radar models still needs to be explained based on the experimental setting;
* It has limited support for open-set unknown radar identification.

---

#### 4.3.3 Radar Emitter Identification under Transfer Learning and Online Learning

**Chinese Title:** 迁移学习与在线学习条件下的雷达辐射源识别
**Publication:** Information, 2020, 11(1):15
**Paper Link:** [Paper](https://www.mdpi.com/2078-2489/11/1/15) / [DOI](https://doi.org/10.3390/info11010015)
**Method Category:** Transfer learning / online learning / SVM / TrAdaBoost / EM-based transductive transfer
**Input Data:** Radar emitter feature samples, with specific features defined by the source-domain and target-domain data
**Label Granularity:** Radar emitter category
**Strict Model Identification?:** Yes. It focuses on domain shift and dynamic updating in radar emitter identification.
**Code Availability:** No official open-source code found

**Summary:**

This paper proposes a radar emitter identification framework that combines transfer learning and online learning to address distribution differences between source and target domains, limited labeled samples in the target domain, and delayed model updates. When only a small number of labeled target-domain samples are available, the paper uses a TrAdaBoost-based SVM model for inductive transfer learning. When no labeled target-domain samples are available, it uses an EM-based transductive transfer-learning method. To improve the model’s adaptation speed to new samples and new environments, an online-learning mechanism is further introduced.

The value of this paper does not lie in proposing a new deep-network architecture, but in being closer to realistic electronic reconnaissance scenarios. Training data often comes from historical libraries or simulation environments, while test data may come from new receiving environments, new SNR conditions, or new target domains. For a model-identification README, this paper is suitable for the directions of “cross-domain generalization, transfer learning, and online updating”.

**Highlights:**

* Explicitly discusses distribution inconsistency between source and target domains;
* Uses TrAdaBoost-SVM when a small number of labeled target-domain samples are available;
* Uses EM-based transductive transfer learning when no target-domain labels are available;
* Introduces online learning to improve real-time model updating;
* Closer to practical deployment issues such as model-library updating and scenario changes.

**Applicable Scenarios:**

* Radar emitter identification with limited target-domain samples;
* Cross-SNR, cross-receiver, and cross-environment model identification;
* Online updating of radar model libraries;
* Surveys or experimental designs emphasizing practical electronic reconnaissance deployment.

**Limitations:**

* The method depends on the source-domain and target-domain sample settings, so domain splits must be clearly defined for reproduction;
* Traditional transfer-learning models have limited ability to represent complex nonlinear sequences;
* No public standard dataset or official code is provided;
* Open-set unknown-class identification still requires additional mechanisms.

---

#### 4.3.4 Radar Emitter Classification with Attribute-specific Recurrent Neural Networks

**Chinese Title:** 基于属性专用循环神经网络的雷达辐射源分类
**Publication:** arXiv / CoRR, 2019, arXiv:1911.07683
**Paper Link:** [arXiv](https://arxiv.org/abs/1911.07683)
**Method Category:** PDW multi-attribute sequence modeling / attribute-specific LSTM / supervised deep classification
**Input Data:** PDW sequences; the paper’s experiments mainly use PRI, PW, RF, and other attributes
**Label Granularity:** Radar emitter category
**Strict Model Identification?:** Relatively yes. It belongs to PDW-sequence-based emitter classification.
**Code Availability:** No official open-source code found

**Summary:**

This paper argues that radar pulse streams have complex temporal patterns, and static numerical analysis of pulse attributes alone is no longer sufficient for stable emitter classification. The method uses recurrent neural networks to model temporal dependencies in pulse sequences and proposes two key designs. First, it performs per-sequence normalization to better exploit the temporal patterns within each pulse sequence. Second, it uses attribute-specific RNN processing, namely building separate recurrent neural network branches for different PDW attributes, allowing features such as RF, PW, and PRI to learn temporal patterns independently in their respective dimensions before fusion and classification.

This method is suitable as a representative of “PDW multivariate temporal modeling”. Compared with directly concatenating all PDW features and feeding them into a single RNN, it emphasizes that different attributes have different variation patterns, making the model structure more consistent with the physical characteristics of radar pulse parameters.

**Highlights:**

* Explicitly treats radar pulse streams as a temporal classification problem;
* Proposes per-sequence normalization, emphasizing dynamic patterns within each sequence;
* Designs independent LSTM branches for different PDW attributes;
* More suitable for modeling heterogeneous multi-attribute variations than RNNs using directly concatenated features;
* Suitable as a deep baseline for multivariate PDW sequence identification.

**Applicable Scenarios:**

* Radar emitter classification based on PDW attributes such as PRI, PW, and RF;
* Multi-pulse sequence-level model identification;
* Model designs that require analyzing the contribution of different parameter dimensions;
* Comparison with Transformers and other multivariate time-series models.

**Limitations:**

* Currently mainly an arXiv preprint, with no formal journal / conference version found;
* Requires labeled training data and cannot serve as an unsupervised method;
* Mainly validated on simulated or constructed data, while real model-identification generalization requires further validation;
* No specific design for open-set unknown emitters.

---

#### 4.3.5 Radar Emitter Classification With Attention-Based Multi-RNNs

**Chinese Title:** 基于注意力多循环神经网络的雷达辐射源分类
**Publication:** IEEE Communications Letters, 2020, 24(9):2000–2004
**Paper Link:** [DOI](https://doi.org/10.1109/LCOMM.2020.2995842) / [PDF](https://www.researchgate.net/publication/341533342_Radar_Emitter_Classification_With_Attention-Based_Multi-RNNs)
**Method Category:** Multi-branch RNN / attention fusion / PDW sequence classification
**Input Data:** Multiple pulse-stream feature sequences, usually PDW parameter sequences
**Label Granularity:** Radar emitter category
**Strict Model Identification?:** Relatively yes. It belongs to radar emitter classification based on pulse-stream features.
**Code Availability:** No official open-source code found

**Summary:**

This paper uses multiple recurrent neural network branches to process different pulse-stream features and applies an attention mechanism to adaptively fuse the outputs of different branches. The core idea is that, in complex electromagnetic environments, different PDW parameters do not contribute equally to emitter classification. For example, PRI may be more discriminative for some radars, while RF or PW may be more discriminative for others. Under noise, missing pulses, or parameter measurement errors, the reliability of different attributes may also change. Therefore, the model needs a mechanism to automatically assign importance to each feature branch.

This method is suitable for the “multi-attribute sequence fusion identification” direction. Compared with attribute-specific RNNs, it not only processes different attributes separately, but also uses an attention mechanism for feature-level or branch-level weighted fusion, emphasizing adaptive selection of feature importance.

**Highlights:**

* Uses multiple RNN branches to separately model different pulse attributes;
* Introduces an attention mechanism to automatically adjust the importance of different feature branches;
* Suitable for radar emitter classification where different parameters have inconsistent discriminative power;
* Has some adaptability to complex pulse streams and noisy conditions;
* Can serve as a representative deep-fusion model for PDW sequence identification.

**Applicable Scenarios:**

* Multi-attribute PDW sequence identification;
* Radar classification where feature reliability changes across scenarios;
* Models requiring interpretation of the contribution of different parameters;
* Comparison with LSTM, GRU, Transformer, and other sequence models.

**Limitations:**

* Still a supervised classification method requiring labeled training data;
* Sensitive to input pulse-stream quality and deinterleaving results;
* No public code is provided, making reproduction relatively costly;
* Cannot directly handle unknown classes or open-set problems.

---

#### 4.3.6 Identification of Radar Emitter Type with Recurrent Neural Networks

**Chinese Title:** 基于循环神经网络的雷达辐射源类型识别
**Publication:** Sensor Signal Processing for Defence Conference, SSPD, 2020
**Paper Link:** [PDF](https://sspd.eng.ed.ac.uk/sites/sspd.eng.ed.ac.uk/files/attachments/basicpage/20200916/S2%20Apfeld.pdf) / [DOI](https://doi.org/10.1109/SSPD47486.2020.9271988)
**Method Category:** LSTM / radar behavior-sequence modeling / multifunction radar type identification
**Input Data:** Multifunction radar emission-behavior sequences or pulse-sequence features
**Label Granularity:** Radar emitter type
**Strict Model Identification?:** Relatively yes. It is more oriented toward radar type / behavior-level identification.
**Code Availability:** No official open-source code found

**Summary:**

This paper uses recurrent neural networks, especially LSTMs, to identify multifunction radar emitter types. Unlike methods that only focus on individual pulse parameters or static statistics, this method emphasizes the implicit temporal grammar in radar emission-behavior sequences. For multifunction radars, different radars may be similar in some individual parameters, but they may show different patterns in search, tracking, dwell behavior, mode switching, and resource management. LSTMs can retain information from historical inputs, making them suitable for modeling such long-term dependencies and state-transition features.

This paper is suitable for the direction of “working-behavior sequence modeling” or “multifunction radar type identification”. It is valuable for this README because it explicitly uses the task expression “Radar Emitter Type”, which is closer to the main line of model / type identification than ordinary modulation classification.

**Highlights:**

* Targets multifunction radar emitter type identification;
* Uses LSTMs to model long-term dependencies in emission-behavior sequences;
* Emphasizes that radar emission sequences have structures similar to “language grammar”;
* Suitable for capturing behavior differences such as search, tracking, dwell, and mode switching;
* Can serve as a representative method for sequence-level identification of multifunction / agile radars.

**Applicable Scenarios:**

* Multifunction radar type identification;
* Behavior-sequence, working-mode-sequence, or dwell-level sequence modeling;
* Scenarios where individual pulse parameters are insufficient and context is needed;
* Radar resource management and intent-inference-related identification tasks.

**Limitations:**

* More oriented toward type identification and behavior modeling, not necessarily specific equipment models;
* Requires a relatively complete temporal observation window;
* Model performance depends on sequence labels and simulation-scenario design;
* Does not directly solve open-set unknown radar identification.

---

#### 4.3.7 Radar Emitter Signal Recognition Based on One-Dimensional Convolutional Neural Network with Attention Mechanism

**Chinese Title:** 基于一维卷积神经网络与注意力机制的雷达辐射源信号识别
**Publication:** Sensors, 2020, 20(21):6350
**Paper Link:** [Paper](https://www.mdpi.com/1424-8220/20/21/6350) / [DOI](https://doi.org/10.3390/s20216350)
**Method Category:** Raw one-dimensional signal recognition / 1D-CNN / attention mechanism / end-to-end deep learning
**Input Data:** One-dimensional radar signal sequences
**Label Granularity:** Radar emitter signal category
**Strict Model Identification?:** Medium to high, depending on whether the data labels correspond to radar models or emitter categories
**Code Availability:** No official open-source code found

**Summary:**

This paper addresses the limitations of traditional radar emitter signal recognition methods, which rely on manual features, consume considerable time, and perform poorly in complex electromagnetic environments. It proposes a one-dimensional convolutional neural network with an attention mechanism, called CNN-1D-AM. The method directly takes one-dimensional time-domain radar signal sequences as input, extracts local discriminative features through 1D convolutional layers, and then uses an attention unit to weight feature maps so that the model focuses more on useful identification information while suppressing redundant or negative features. The paper conducts experiments on seven types of radar emitter signals, showing that the proposed method outperforms several comparison methods in recognition accuracy and performance.

This method differs from PDW sequence identification and is more oriented toward intra-pulse or raw-signal end-to-end recognition. It is suitable for the “raw signal / one-dimensional signal recognition” category in a model-identification README. If the labels are indeed emitter categories, it can be considered a model-identification-related method. However, if the labels are only waveform or modulation categories, the task boundary must be explained carefully.

**Highlights:**

* Directly processes one-dimensional radar signal sequences without converting them into two-dimensional images;
* Uses 1D-CNN to automatically extract local time-domain features;
* The attention mechanism assigns higher weights to key features;
* Compared with 2D CNNs, it reduces the time and storage cost caused by dimensional conversion;
* Suitable as an intra-pulse signal branch or end-to-end identification baseline.

**Applicable Scenarios:**

* End-to-end identification of one-dimensional radar signals;
* Intra-pulse sampled-data classification;
* Scenarios with real-time requirements where time-frequency images are not desired;
* Comparison with time-frequency-image CNNs and PDW sequence models.

**Limitations:**

* Does not directly model inter-pulse behavior such as multi-pulse PRI, RF, and PW patterns;
* If the data labels are not models or emitter categories, it cannot be directly called model identification;
* Cross-domain generalization across receivers, SNRs, and real scenarios still requires validation;
* No official open-source code is provided.

---

#### 4.3.8 A Knowledge Graph-Driven CNN for Radar Emitter Identification

**Chinese Title:** 知识图谱驱动的卷积神经网络雷达辐射源识别
**Publication:** Remote Sensing, 2023, 15(13):3289
**Paper Link:** [Paper](https://www.mdpi.com/2072-4292/15/13/3289) / [DOI](https://doi.org/10.3390/rs15133289)
**Method Category:** Knowledge graph / 1D-CNN / knowledge-enhanced radar emitter identification
**Input Data:** Intermediate-frequency radar data or radar sample data, combined with a knowledge graph to construct task-related precise datasets
**Label Granularity:** Specific emitter individual or radar emitter category
**Strict Model Identification?:** Highly related. The paper is more oriented toward specific emitter identification, and the granularity may be finer than model identification.
**Code Availability:** No official open-source code found

**Summary:**

This paper introduces knowledge graphs into radar emitter identification and proposes a KG-1D-CNN framework. Traditional intelligent identification methods usually focus on network structures, feature extraction, or classification accuracy, while paying less attention to the relationships between radar data and task-scenario information. This method first constructs a radar emitter knowledge graph, using nodes and edges to represent relationships among radar data, and uses an intelligent recognizer to measure recognition difficulty and association among samples. It then selects and organizes precise datasets suitable for specific tasks according to the knowledge graph, and finally uses a one-dimensional convolutional neural network to identify intermediate-frequency radar data.

The value of this direction lies in introducing expert knowledge, task information, and data relationships from electronic reconnaissance into the training process of deep models. It is not merely end-to-end classification, but emphasizes a pipeline of “knowledge organization + data selection + deep identification”. Therefore, it is suitable for the category of “knowledge-enhanced model identification” or “knowledge-graph-assisted radar identification”.

**Highlights:**

* Constructs a radar emitter knowledge graph to describe relationships among radar data;
* Uses the knowledge graph to select precise training data suitable for the task;
* Combines the knowledge graph with 1D-CNN for radar intermediate-frequency data identification;
* Emphasizes prior knowledge, data organization, and interpretability in identification tasks;
* Suitable for connecting traditional threat libraries, expert systems, and deep-learning methods.

**Applicable Scenarios:**

* Knowledge-enhanced radar emitter identification;
* Model-identification systems that need to fuse expert knowledge and deep learning;
* Identification tasks with clear task scenarios and usable historical data relationships;
* Specific emitter identification or fine-grained radar identification.

**Limitations:**

* More oriented toward specific emitter identification, and the granularity may be finer than model identification;
* Knowledge-graph construction depends on existing data and expert rules;
* Reproduction requires a relatively complete radar sample library and task scenario;
* No official open-source code is provided.

---

#### 4.3.9 A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and TFCN

**Chinese Title:** 基于层次聚类和时频卷积网络的多雷达辐射源分选与识别方法
**Publication:** Digital Signal Processing, 2025, Volume 160, Article 105005
**Paper Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1051200425000272) / [DOI](https://doi.org/10.1016/j.dsp.2025.105005)
**Method Category:** Joint sorting-recognition pipeline / hierarchical clustering / KDE-KLD-TM / time-frequency convolutional network, TFCN
**Input Data:** Interleaved PDW pulse streams, with time-frequency representations further constructed after sorting for recognition
**Label Granularity:** Multi-radar emitter categories
**Strict Model Identification?:** Medium to high. The method includes both sorting and recognition, and the recognition part is related to emitter categories.
**Code Availability:** No official open-source code found. The paper page states that the authors do not have permission to share the data.

**Summary:**

This paper jointly handles sorting and recognition in interleaved multi-radar pulse streams. To address the high complexity of existing sorting methods and the difficulty of intra-class clustering and inter-class interleaving, the paper proposes a hierarchical clustering sorting method based on KDE-KLD-TM. The method first uses downsampled data to construct central clusters, improving processing speed. It then performs full-sample inter-class clustering based on maximum a posteriori probability. Finally, it combines cluster-distribution similarity and periodic template matching to complete intra-class merging and inter-class deinterleaving. After sorting, the paper further proposes a PDW-based time-frequency convolutional network, TFCN, which converts a one-dimensional sequence classification problem into a two-dimensional image classification problem. It extracts periodic features through time-frequency analysis and uses a CNN for emitter identification.

The feature of this method is that it is closer to a complete electronic reconnaissance pipeline: first separating different emitters from interleaved pulse streams, and then identifying the sorted results. For a model-identification resource list, it is suitable for the direction of “joint sorting-recognition modeling” rather than a simple classification model.

**Highlights:**

* Considers both multi-radar pulse-stream sorting and emitter identification;
* Uses KDE-KLD-TM hierarchical clustering to handle intra-class clustering and inter-class interleaving;
* Constructs central clusters through downsampling, improving large-scale interleaved-sample processing speed;
* Proposes a TFCN identification method from the PDW perspective, emphasizing PRI-period differences;
* Converts one-dimensional sequence identification into two-dimensional time-frequency image classification;
* The paper reports processing 2.06 million samples in typical interleaving scenarios with high recognition accuracy.

**Applicable Scenarios:**

* Multi-radar interleaved pulse-stream processing;
* Joint sorting and recognition experiments;
* Complex scenarios involving intra-class clustering, inter-class interleaving, and pulse loss;
* Research combining traditional clustering-based sorting with deep recognition networks.

**Limitations:**

* The method pipeline is complex, including clustering, template matching, time-frequency analysis, and CNN;
* The data is not public, making reproduction difficult;
* Recognition performance depends on the quality of the front-end sorting process;
* If the task only studies clean single-emitter sequence model identification, this method may be overly complex.

---

### 4.4 Method Category Comparison Table

| Method / Paper                                                                                                | Method Category                          | Input Form                                         | Label Granularity             | Strict Model Identification? | Code / Data Status       | Recommended Use                              |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------- | ----------------------------- | ---------------------------- | ------------------------ | -------------------------------------------- |
| Radar Emitter Recognition Based on Parameter Set Clustering and Classification                                | Parameter-set clustering + decision tree | PDW / pulse train                                  | Radar emitter category        | Yes                          | No official code         | Traditional PDW parameter-set baseline       |
| Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism | Dynamic CNN + LSTM + attention           | Deinterleaved PDW sequence                         | Agile-waveform radar category | Yes                          | GitHub project available | Deep PDW sequence identification baseline    |
| Radar Emitter Identification under Transfer Learning and Online Learning                                      | Transfer learning + online learning      | Source-domain / target-domain features             | Radar emitter category        | Yes                          | No official code         | Cross-domain, few-shot, online updating      |
| Radar Emitter Classification with Attribute-specific RNNs                                                     | Attribute-specific LSTM                  | Multi-attribute PDW sequence                       | Radar emitter category        | Yes                          | No official code         | Multivariate temporal modeling               |
| Radar Emitter Classification With Attention-Based Multi-RNNs                                                  | Multi-RNN + attention                    | Multi-attribute PDW sequence                       | Radar emitter category        | Yes                          | No official code         | Multi-branch feature fusion                  |
| Identification of Radar Emitter Type with RNNs                                                                | LSTM behavior-sequence modeling          | Emission behavior / pulse sequence                 | Radar type                    | Yes                          | No official code         | Multifunction radar behavior identification  |
| Radar Emitter Signal Recognition Based on 1D-CNN-AM                                                           | 1D-CNN + attention                       | One-dimensional radar signal                       | Signal / emitter category     | Depends on labels            | No official code         | End-to-end intra-pulse signal identification |
| A Knowledge Graph-Driven CNN for Radar Emitter Identification                                                 | Knowledge graph + 1D-CNN                 | Intermediate-frequency data / radar sample library | Individual or category        | Highly related               | No official code         | Knowledge-enhanced identification            |
| A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and TFCN                | Hierarchical clustering + TFCN           | Interleaved PDW pulse stream                       | Multi-radar category          | Medium to high               | Data not public          | Joint sorting-recognition pipeline           |

---

## 5. Recommended Experimental Settings

### 5.1 Suggested Baselines

| Category                     | Method                                                                                        | Use                                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Parameter-matching baseline  | Parameter-template matching, expert-rule libraries, nearest-neighbor template matching        | Engineering-style model-identification baseline        |
| Traditional machine learning | SVM, random forest, gradient boosting trees, KNN, Gaussian mixture model, hidden Markov model | PDW / parameter-feature classification                 |
| Sequence modeling            | HMM, LSTM, GRU, Transformer encoder                                                           | PDW sequence and working-mode sequence identification  |
| Time-frequency methods       | STFT / CWT / pseudo Wigner-Ville distribution + CNN                                           | Intra-pulse structure-aided identification             |
| Raw-signal methods           | 1D CNN, temporal convolutional network, Transformer                                           | End-to-end IQ or one-dimensional signal identification |
| Transfer learning            | TrAdaBoost, domain adaptation, online learning                                                | Few-shot target-domain and online updating             |
| Open-set identification      | OpenMax, energy score, prototype distance, metric autoencoder                                 | Unknown emitter detection                              |
| Representation learning      | Autoencoder, contrastive learning, masked signal modeling                                     | Pretraining and low-label identification               |

---

### 5.2 Suggested Pipelines

If using PDW / parameter-set input:

```text
PDW sequence / parameter set
        ↓
Feature cleaning and normalization
        ↓
Parameter statistics / temporal features / working-mode features
        ↓
Traditional classifier or deep sequence model
        ↓
Radar model / functional category / unknown-class decision
```

If using IQ or time-frequency image input:

```text
Raw IQ signal
        ↓
Preprocessing and normalization
        ↓
STFT / CWT / pseudo Wigner-Ville distribution or end-to-end encoder
        ↓
CNN / 1D-CNN / Transformer
        ↓
Model category / waveform auxiliary category / unknown-class decision
```

---

## 6. Notes

* The core task of this page is radar emitter model identification, not radar pulse deinterleaving, waveform recognition, modulation recognition, or specific emitter identification;
* Strict model identification requires labels corresponding to radar models, equipment families, functional categories, or known emitter classes;
* Public real radar model-level data is very limited, and related datasets must be clearly described as substitute tasks or auxiliary resources when used;
* Modulation categories, waveform categories, and hardware individual IDs cannot be directly equated with radar models;
* When using simulated model libraries, the RF, PRI, PW, modulation type, working mode, and parameter distribution of each model should be clearly specified;
* When reporting results, it is not recommended to report only accuracy. Macro F1, confusion matrices, cross-domain results, and open-set metrics should also be provided;
* Open-set identification, cross-domain generalization, and online updating are closer to real electronic reconnaissance scenarios than ordinary closed-set classification;
* When real radar signal data is involved, data sources, licenses, safety compliance, and confidentiality requirements must be followed.

---

## 7. Related Contributions and Patent Layout from Our Lab

In addition to public papers and open-source methods, our lab has also conducted systematic research on radar emitter model identification and has developed several patent achievements for non-cooperative radar emitter identification. These works mainly focus on contextual semantic modeling, multivariate temporal modeling, self-supervised representation learning, and intra-pulse / inter-pulse joint contrastive learning. They complement existing PDW sequence identification, IQ signal identification, and open-set identification methods.

| No. | Patent / Method Name                                                                                                  | Country / Region | Method Direction                                     | Main Contribution                                                                                                                                                                                                                                  | Relationship to Radar Model Identification                                                                                                                              |
| --: | --------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | A radar emitter model identification method based on context-aware semantic modeling                                  | China            | Contextual semantic modeling                         | Mines contextual semantic relationships from radar pulse sequences or emitter behavior sequences, using associations among preceding and following pulses, parameter changes, and operational behaviors to enhance model-identification capability | Suitable for multifunction or agile radar scenarios where a single pulse does not provide sufficient discriminative information and sequence context is required        |
|   2 | A radar emitter model identification method based on multivariate temporal Transformer                                | China            | Multivariate temporal Transformer                    | Treats PDW parameters such as RF/CF, PW, PRI, PA, and DOA/AOA as multivariate temporal inputs, and uses Transformers to model long-range dependencies and cross-variable interactions                                                              | Suitable for PDW-sequence-based radar emitter model identification, and can serve as an enhanced alternative to RNN / LSTM methods                                      |
|   3 | A non-cooperative radar emitter model identification method based on hybrid self-supervised learning                  | China            | Self-supervised representation learning              | Addresses the shortage of labeled samples in non-cooperative scenarios by learning general representations from unlabeled radar signals or PDW data through self-supervised tasks, and then transferring them to model-identification tasks        | Suitable for real electronic reconnaissance environments with scarce labels, large scenario variation, and insufficient samples from new emitters                       |
|   4 | A non-cooperative radar emitter model identification method based on intra-pulse and inter-pulse contrastive learning | China            | Intra-pulse / inter-pulse joint contrastive learning | Uses both intra-pulse signal features and inter-pulse PDW sequence features, and enhances the consistency of same-class emitter samples and separability of different emitter samples through contrastive learning                                 | Suitable for multimodal radar emitter model identification, integrating IQ / intermediate-frequency intra-pulse information with PDW inter-pulse behavioral information |

---

## 8. Citation and Contribution

This resource list is organized and maintained by the **SmartDSP Lab, School of Informatics, Xiamen University**. It aims to provide a relatively systematic, reproducible, and extensible open-source resource index for radar model identification and related intelligent electromagnetic-signal processing research.

Contributions and corrections are welcome via Issues or Pull Requests, including but not limited to:

* Newly released radar emitter model identification datasets;
* Public resources with clear labels corresponding to radar models, functional categories, or emitter classes;
* Papers related to radar emitter identification, open-set identification, cross-domain generalization, and online learning;
* Reproducible benchmark results for model identification;
* Additional notes on the boundaries between modulation recognition, waveform recognition, specific emitter identification, and model identification.

When submitting a resource, it is recommended to include:

```text
Resource name:
Resource link:
Task type:
Label meaning:
Whether it strictly belongs to model identification:
Whether the data is open-source:
Whether the code is open-source:
Reason for recommendation:
Notes:
```

This repository is only intended for public academic resource organization and reproducible experiment reference. It does not provide non-public radar databases, does not provide sensitive system parameters, and does not support any unauthorized signal interception, identification, or electronic-countermeasure use.
