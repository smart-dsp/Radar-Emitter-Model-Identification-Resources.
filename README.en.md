# Radar Emitter Model Identification Resources

<p align="center">
  <a href="./README.md">简体中文</a> |
  <b>English</b>
</p>

This repository is maintained by the **SmartDSP Lab, School of Informatics, Xiamen University**. It collects public resources, representative papers, and reproducible experiment suggestions related to **Radar Emitter Model Identification / Radar Emitter Type Recognition / Radar Emitter Classification**.

The core task discussed here is **radar emitter model identification**, namely identifying which **known radar model, equipment family, functional category, or emitter class** a captured radar signal belongs to, based on intercepted radar signals, Pulse Description Word (PDW) sequences, intra-pulse features, or radar behavior sequences.

This resource list follows the principles below:

* Prioritize papers and resources directly related to **radar emitter recognition, radar emitter model identification, and radar emitter type recognition**;
* Treat datasets with only modulation labels, waveform labels, or device-level individual IDs as related tasks or auxiliary resources;
* Exclude projects without clear papers, data descriptions, or reproducible value from the main recommendations;
* Put highly relevant methods without public data or official code under “Representative Papers” rather than “Open-source Resources”.

---

## Key Takeaways

* **Public datasets for strict radar model identification are extremely scarce.** There is currently no stable, public, and reproducible real-world radar model-level benchmark. Most papers use simulated PDWs, simulated radar parameter sets, internal intercepted data, or self-built model libraries.
* **The closest public resources** are datasets such as RadarCommDataset, which contain radar functional categories or signal-type labels. However, they are still not strict radar model-level datasets.
* **Representative papers** should focus on PDW parameter-set recognition, radar emitter recognition, open-set radar emitter recognition, transfer learning / online learning, and agile-waveform radar emitter recognition.
* **The label semantics must be clearly stated in any experiment report.** Whether the labels correspond to radar models, functional categories, waveform classes, modulation classes, individual device IDs, or work modes determines whether the task is truly radar model identification.

---

## Table of Contents

* [Key Takeaways](#key-takeaways)
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
  * [3.2 Notes on Model-identification Data](#32-notes-on-model-identification-data)

    * [3.2.1 Real Radar Model-level Datasets](#321-real-radar-model-level-datasets)
    * [3.2.2 RadarCommDataset](#322-radarcommdataset)
    * [3.2.3 Simulated Radar Model Libraries](#323-simulated-radar-model-libraries)
* [4. Representative Methods and Papers](#4-representative-methods-and-papers)

  * [4.1 Paper Overview](#41-paper-overview)
  * [4.2 Method Summaries](#42-method-summaries)
* [5. Recommended Experimental Settings](#5-recommended-experimental-settings)

  * [5.1 Suggested Baselines](#51-suggested-baselines)
  * [5.2 Suggested Pipelines](#52-suggested-pipelines)
* [6. Notes](#6-notes)
* [7. Lab Contributions and Patent Portfolio](#7-lab-contributions-and-patent-portfolio)
* [8. Citation and Contribution](#8-citation-and-contribution)

---

## 1. Task Overview

### 1.1 What Is Radar Emitter Model Identification?

Radar emitter model identification refers to identifying the **radar model, equipment family, functional category, operating regime, known emitter class, or unknown class** of a target emitter based on radar emissions captured by electronic intelligence, radar warning, signal intelligence, or electromagnetic spectrum monitoring systems. It is not a single fixed classification problem. Instead, it can be divided into multiple forms according to the input data format, sample granularity, and label granularity.

In terms of input data, radar emitter model identification can be based on **PDW parameters**, **intermediate-frequency signals, baseband IQ signals, intra-pulse sampled data, time-frequency images, spectra, or multimodal fused features**. PDW data are usually obtained from parameter measurements of pulse signals by electronic reconnaissance receivers and include time of arrival, pulse width, carrier frequency, amplitude, direction of arrival, and other parameters. IQ or intermediate-frequency data preserve richer intra-pulse modulation, frequency variation, phase variation, and waveform-structure information.

In terms of sample granularity, model identification can be performed at the **single-pulse level** or the **multi-pulse sequence level**. Single-pulse identification uses one pulse or a short segment of intra-pulse samples as input and predicts the possible emitter class of that pulse. Multi-pulse sequence identification uses a continuous pulse sequence, an emitter track, or an intercepted time slice as input and exploits PRI variation, frequency agility, pulse-width variation, amplitude variation, direction-of-arrival variation, and behavioral patterns to identify the emitter class corresponding to the sequence. For agile radars, multifunction radars, and modern cognitive radars, sequence-level modeling often better reflects radar behavioral characteristics than single-pulse classification.

In terms of output labels, identification results can have different levels of granularity. Coarse-grained labels may represent **radar functional categories**, such as search radar, tracking radar, altimeter radar, airborne detection radar, or ground-surveillance radar. Medium-grained labels may represent **radar regimes or equipment families**, such as pulse-Doppler radar, phased-array radar, frequency-modulated continuous-wave radar, or a specific equipment family. Fine-grained labels may represent **specific radar models or known emitter classes**. In open-set scenarios, the model should also determine whether an input belongs to an unknown emitter that was not present in the training set.

A typical radar emitter model identification pipeline can be described as:

```text
Intercepted radar signal / PDW data / IF signal / IQ signal
        ↓
Preprocessing, pulse detection, parameter measurement, or time-frequency transform
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

where `yi` may denote the predicted known radar category, emitter class, or an intermediate result used by subsequent sequence-level identification.

If the input is a multi-pulse PDW sequence, one sample can be represented as:

```text
P = [p1, p2, ..., pT]
```

where each pulse `pt` may contain:

```text
pt = [TOA, PRI, RF/CF, PW, PA, DOA/AOA, ...]
```

The model then learns a mapping from a pulse sequence to an emitter class:

```text
f(P) -> y
```

where `y` usually denotes the radar model, radar type, or known emitter class corresponding to the sequence, emitter track, or intercepted time slice.

If the input is an intermediate-frequency signal or a baseband IQ signal, one sample can be represented as:

```text
x = [I1 + jQ1, I2 + jQ2, ..., IN + jQN]
```

or as an intermediate-frequency sampled sequence:

```text
s = [s1, s2, ..., sN]
```

In this case, the model can learn intra-pulse modulation, frequency variation, phase variation, envelope structure, and spectral features directly from the raw waveform:

```text
f(x) -> y
```

In more complex scenarios, the model may jointly use PDW parameters and IQ / intermediate-frequency data for multimodal fusion:

```text
f(P, x) -> y
```

Here, the PDW sequence provides inter-pulse behavioral features, while IQ or intermediate-frequency signals provide intra-pulse waveform details. The former is better suited to describing radar operating patterns and parameter agility, whereas the latter is better suited to capturing intra-pulse modulation, hardware characteristics, and waveform details. Practical radar emitter model identification systems often need to combine both sources of information to achieve stable performance in complex electromagnetic environments, low-SNR conditions, multi-emitter mixtures, and unknown-target scenarios.

Therefore, radar emitter model identification should be understood as a multi-granularity, multimodal, and multi-level recognition task. It can be single-pulse classification or pulse-sequence classification; it can be based on PDW parameters, intermediate-frequency signals, IQ signals, spectra, or time-frequency images; and the output may be the category of a single pulse, a pulse sequence, an emitter track, a time slice, or multiple radar emitters present in a scene.

### 1.2 Differences from Related Tasks

| Task                                                | Output                                                                                    | Relationship to Model Identification                         | Equivalent to Model Identification? |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| Radar pulse deinterleaving                          | The emitter cluster to which each pulse belongs                                           | A preprocessing or parallel task before model identification | No                                  |
| Radar waveform / intra-pulse modulation recognition | LFM, Barker, Frank, P1-P4, Costas, and other waveform or modulation classes               | Useful evidence for model identification                     | No                                  |
| Radar functional category recognition               | Search, tracking, altimetry, imaging, airborne detection, and other functional categories | Can be viewed as coarse-grained model identification         | Partially                           |
| Radar emitter model identification                  | Radar model, equipment family, known emitter class                                        | The core task of this page                                   | Yes                                 |
| Specific emitter identification                     | A specific device or individual transmitter ID                                            | Finer-grained than model identification                      | No                                  |
| RF fingerprinting                                   | Hardware identity of an individual transmitter                                            | Can provide methods for radar individual identification      | No                                  |
| Radar work-mode recognition                         | Search, tracking, guidance, altimetry, and other modes                                    | Can support model identification and intent inference        | No                                  |
| Radar spectrum detection                            | Whether radar signals exist in the spectrum and their frequency positions                 | A detection task rather than model classification            | No                                  |
| Radar jamming recognition                           | Jamming pattern or jamming class                                                          | An electronic-countermeasure-related task                    | No                                  |

---

### 1.3 Common Input Features

| Feature or Representation       | Meaning                                                                                                  | Usage                                                                             |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Raw IQ                          | Complex baseband sampled signal                                                                          | End-to-end 1D CNN, Transformer, self-supervised pretraining                       |
| Time-frequency image            | STFT, CWT, Wigner-Ville distribution, pseudo Wigner-Ville distribution, Choi-Williams distribution, etc. | Intra-pulse structure, modulation pattern, waveform-regime recognition            |
| TOA                             | Time of arrival                                                                                          | PRI estimation and mode-sequence analysis                                         |
| PRI / DTOA                      | Pulse repetition interval / difference of time of arrival                                                | Distinguishing repetition periods and PRI modulation patterns                     |
| RF / CF                         | Radio frequency / carrier frequency                                                                      | Recognizing carrier frequency, frequency hopping, and frequency agility           |
| PW                              | Pulse width                                                                                              | Distinguishing pulse parameters and emission regimes                              |
| PA / AMP                        | Pulse amplitude                                                                                          | Auxiliary evidence, but affected by propagation path and receiver characteristics |
| DOA / AOA                       | Direction of arrival / angle of arrival                                                                  | Spatial auxiliary features in multi-emitter scenarios                             |
| Intra-pulse modulation features | LFM, NLFM, BPSK, Barker, Frank, P1-P4, etc.                                                              | Useful for inferring radar regimes, but not equivalent to radar models            |
| Work-mode sequence              | Transition sequence among search, tracking, guidance, and other modes                                    | Work-mode recognition, model-assisted identification, intent inference            |
| Hardware fingerprint features   | Non-ideal characteristics of the transmitter                                                             | Specific emitter identification                                                   |

---

### 1.4 Common Evaluation Metrics

| Metric                | Description                                                      | Recommended Scenario                                            |
| --------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------- |
| Accuracy              | Classification accuracy                                          | Closed-set identification with balanced classes                 |
| Balanced Accuracy     | Average recall over all classes                                  | Class-imbalanced data                                           |
| Macro-F1              | Macro average of F1 scores over all classes                      | Multi-class imbalanced identification                           |
| Weighted-F1           | F1 score weighted by the number of samples in each class         | Datasets with large class-frequency differences                 |
| Confusion Matrix      | Displays class-to-class confusion                                | Analyzing easily confused radar models or functional categories |
| Top-k Accuracy        | Whether the correct class appears among the top-k predictions    | Large model libraries                                           |
| AUROC                 | Ability to separate known and unknown classes                    | Open-set / out-of-distribution recognition                      |
| AUPR                  | Detection ability under imbalanced positive and negative samples | Open-set / out-of-distribution recognition                      |
| FPR95                 | False positive rate when true positive rate is 95%               | Open-set recognition                                            |
| ECE                   | Expected calibration error                                       | Reliability evaluation                                          |
| Cross-domain Accuracy | Accuracy across SNRs, receivers, or scenarios                    | Generalization evaluation                                       |
| Few-shot Accuracy     | Accuracy under N-way K-shot settings                             | Few-shot model identification                                   |

---

## 2. Method Taxonomy and Task Boundaries

### 2.1 Methods for Strict Model Identification

The following methods directly target radar models, radar types, functional categories, or known emitter classes.

| Method Type                                 | Task Relevance | Description                                                                                                                   |
| ------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Parameter-library / threat-library matching | High           | Match RF, PRI, PW, scan period, work modes, and other parameters against a known radar library                                |
| Expert systems / rule-based reasoning       | High           | Identify radar types or functional categories using electronic-intelligence expert rules                                      |
| PDW parameter-set classification            | High           | Use RF, PRI, PW, PA, DOA, and other PDW features for emitter-class identification                                             |
| Traditional machine-learning classification | High           | Use SVM, random forest, gradient-boosted trees, k-nearest neighbors, etc. for parameter or statistical feature classification |
| Work-mode sequence modeling                 | High           | Use HMM, LSTM, Transformer, etc. for mode-sequence and multi-pulse behavior recognition                                       |
| Transfer learning / online learning         | High           | Address new scenarios, few-shot samples, and dynamic updates of model libraries                                               |
| Open-set model identification               | High           | Identify unknown emitters or unknown models at test time                                                                      |
| Multi-domain feature fusion                 | High           | Fuse PDW, intra-pulse modulation, time-frequency features, and work-mode features for model identification                    |

---

### 2.2 Related but Non-equivalent Methods

The following methods are highly related to model identification but should not be directly described as strict radar model identification.

| Method Type                                                    | Relationship to Model Identification   | Description                                                                                                             |
| -------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Radar waveform / modulation recognition                        | Auxiliary task                         | Waveform classes can serve as model-identification cues, but a waveform class alone is not a radar model                |
| Radar functional category recognition                          | Coarse-grained related task            | If the label is a radar functional category, it can be viewed as type recognition but not specific model identification |
| Specific emitter identification / RF fingerprinting            | Finer-grained related task             | Identifies individual devices, usually at a finer granularity than model identification                                 |
| Radar spectrum detection                                       | Preceding detection task               | Determines whether radar signals exist in the spectrum; does not output models                                          |
| Radar jamming recognition                                      | Related electronic-countermeasure task | Recognizes jamming patterns rather than radar emitter models                                                            |
| CNN / Vision Transformer on modulation-classification datasets | Auxiliary method                       | Can be used for pretraining or waveform recognition, but not as the main model-identification task                      |

---

### 2.3 Relationship Between Methods and Task Definitions

| Method Family                                       | Suitable for Model Identification? | Label Requirement                                             | Description                                                        |
| --------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------ |
| Parameter-library / threat-library matching         | Yes                                | Radar model, functional category, or equipment-family labels  | Classical engineering approach, but depends on a complete database |
| PDW parameter-set classification                    | Yes                                | Emitter-class labels                                          | Closest to radar emitter recognition in electronic intelligence    |
| Traditional machine learning + PDW features         | Yes                                | Emitter-class labels                                          | Recommended as a basic baseline                                    |
| Deep networks + PDW / IQ / time-frequency features  | Yes                                | Emitter-class labels                                          | Suitable for modeling complex nonlinear features                   |
| Transfer learning / online learning                 | Yes                                | Source-domain or partially labeled target-domain class labels | Suitable for scenario changes and model-library updates            |
| Open-set recognition                                | Yes                                | Known-class labels; unknown classes for testing or validation | Closer to realistic electronic-intelligence scenarios              |
| Waveform / modulation recognition                   | Partially related                  | Waveform or modulation labels                                 | Should not be directly equated with model identification           |
| Specific emitter identification / RF fingerprinting | Partially related                  | Device-level individual IDs                                   | Should not be directly equated with model identification           |
| Spectrum detection / jamming recognition            | Related task                       | Detection labels or jamming labels                            | Not model identification                                           |

---

## 3. Datasets and Public Resources

### 3.1 Dataset Overview

| Dataset / Resource              | Label Semantics                                                                  | Strict Model Identification? | Recommendation | Link                                                                      | Notes                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------- | ---------------------------- | -------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Real radar model-level datasets | Radar model / equipment family / known emitter class                             | Yes                          | ⭐              | No stable public benchmark yet                                            | Real data are sensitive, and publicly reproducible resources are extremely scarce                               |
| RadarCommDataset                | Radar / communication signal classes, including some radar functional categories | Partially                    | ⭐⭐⭐            | [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset) | Suitable as a starting point for functional-category recognition or radar / communication signal classification |

---

### 3.2 Notes on Model-identification Data

#### 3.2.1 Real Radar Model-level Datasets

Strict radar emitter model identification requires labels corresponding to real radar models, equipment families, functional categories, or known emitter classes. Because real radar signals are often sensitive, publicly downloadable and reproducible datasets with clear model-level labels are extremely rare.

Therefore, this repository does not list any real radar model-level public dataset as a main benchmark. If the research target is strict model identification, we recommend using a self-built simulated model library or internal compliant data, with a clear description of label design and simulation parameters.

#### 3.2.2 RadarCommDataset

RadarCommDataset is one of the public resources closest to “radar functional-category recognition / radar and communication signal classification”. It was released by ANDRO’s MR Lab to support single-task and multi-task learning for wireless signals. The dataset contains both radar and communication waveforms and provides signal-class labels, modulation-class labels, SNR values, and IQ samples.

It should be noted that RadarCommDataset is **not a strict real radar model-level dataset**. Its labels are not specific radar models or equipment families, but signal types and functional categories. Therefore, this repository treats it as a **public proxy dataset for radar functional-category recognition / radar-communication signal classification**. It is suitable for evaluating a model’s ability to classify radar functional signals and for building basic deep-learning baselines, but it should not be described as a real radar emitter model-identification benchmark.

**Data content:**

* The data format is HDF5;
* Each sample contains fields such as `modulation`, `signal`, `snr`, and `sample`;
* The IQ input is a 256-dimensional tensor containing 128 I samples and 128 Q samples;
* The sampling rate is 10 MS/s;
* Each waveform contains multiple snapshot samples;
* The SNR range covers -20 dB to 18 dB with a step of 2 dB;
* The dataset provides synthetic dynamic-channel, AWGN-only, and over-the-air 2.45 GHz versions.

**Label types:**

* Modulation classes include pulsed continuous wave, frequency-modulated continuous wave, BPSK, AM-DSB, AM-SSB, ASK, etc.;
* Signal classes include airborne detection radar, airborne range radar, air-ground moving target indication radar, ground mapping radar, radar altimeter, SATCOM, AM radio, short-range wireless, etc.

From the perspective of model identification, RadarCommDataset is useful in the following ways:

1. **It can serve as a starting point for radar functional-category recognition.** The dataset contains several radar functional signal classes, such as airborne detection radar, range radar, air-ground moving target indication radar, ground mapping radar, and radar altimeter. It is suitable for coarse-grained radar type / functional-category recognition experiments.
2. **It can be used to distinguish radar and communication signals.** Since the dataset includes both radar and communication signals, it can support preliminary experiments on “radar signal detection + radar functional-category recognition”.
3. **It is suitable for cross-SNR robustness experiments.** The dataset provides multiple SNR conditions, enabling evaluation of classification performance under low-SNR conditions.
4. **It is suitable for multi-task learning.** Each sample has both signal-class and modulation-class labels, making it useful for studying joint learning between signal-category recognition and modulation recognition.
5. **It is suitable for building public baselines.** The repository provides data-loading and visualization scripts, making it convenient to build basic models such as 1D CNN, temporal convolutional networks, Transformer, and 1D ResNet.

**Usage boundaries:**

* It can be described as radar functional-category recognition, radar / communication signal classification, or a related proxy task;
* It should not be described as a real radar model-level identification dataset;
* The modulation label should not be treated as a radar model label;
* If it is used in a model-identification paper, the label granularity should be explicitly stated as functional category and signal type rather than specific equipment model;
* If it is combined with a self-built simulated model library or internal measured data, RadarCommDataset should be used as pretraining data, auxiliary validation, or a public baseline, not as the final model-level evaluation benchmark.

**Links:** [GitHub](https://github.com/ANDROComputationalSolutions/RadarCommDataset) / [Paper](https://doi.org/10.1016/j.comnet.2021.108441)

#### 3.2.3 Simulated Radar Model Libraries

In the absence of public real model-level data, a reasonable approach is to build a simulated radar model library. Each “model” should not be defined merely by a single waveform class, but by multiple parameters, for example:

```text
Radar model = {
  RF / CF distribution,
  PRI type and parameters,
  PW range,
  intra-pulse modulation,
  work-mode sequence,
  scan pattern,
  parameter-agility pattern
}
```

This design is closer to model identification than simple modulation classification. Experiments should clearly report the parameter ranges, inter-class differences, intra-class variations, training-test splits, and whether distribution shifts exist.

---

## 4. Representative Methods and Papers

This section only includes papers directly related to **radar emitter recognition / radar model identification / radar type recognition / open-set radar emitter recognition**. Papers on modulation recognition, specific emitter identification, spectrum detection, and jamming recognition are not included in the main table.

### 4.1 Paper Overview

| Method / Direction                           | Representative Paper                                                                                                         | Link                                                                                                                                                                       | Code / Project                                           | Task Relevance | Notes                                                                                                                                                                                                                                      |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Parameter-set clustering and classification  | Radar Emitter Recognition Based on Parameter Set Clustering and Classification                                               | [Paper](https://www.mdpi.com/2072-4292/14/18/4468) / [DOI](https://doi.org/10.3390/rs14184468)                                                                             | Paper method                                             | High           | Uses PDW parameter-set clustering, frequent-item construction, and decision-tree classification for radar emitter recognition. The task is close to model / type identification and suitable as a traditional machine-learning baseline    |
| Agile-waveform emitter recognition           | Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism                | [PDF](https://www.radioeng.cz/fulltexts/2021/21_04_0704_0712.pdf)                                                                                                          | [GitHub](https://github.com/fengyuntian2009/PDWSequence) | High           | Uses deinterleaved PDW sequences and combines dynamic CNN, LSTM, and attention to identify agile-waveform radar emitter types. This is one of the closest directions with an open-source project                                           |
| Transfer learning and online learning        | Radar Emitter Identification under Transfer Learning and Online Learning                                                     | [Paper](https://www.mdpi.com/2078-2489/11/1/15) / [DOI](https://doi.org/10.3390/info11010015)                                                                              | Paper method                                             | High           | Targets radar emitter identification under limited target-domain samples, environmental changes, and dynamic updates; uses transfer learning, TrAdaBoost-SVM, expectation-maximization, and online learning                                |
| Attribute-specific RNN sequence modeling     | Radar Emitter Classification with Attribute-specific Recurrent Neural Networks                                               | [arXiv](https://arxiv.org/abs/1911.07683)                                                                                                                                  | Paper method                                             | High           | Models different attributes in PDW pulse streams separately and fuses temporal features. Suitable for emitter type recognition based on RF, PW, PRI, AOA, PA, and other multivariate sequences                                             |
| Attention-based Multi-RNN                    | Radar Emitter Classification With Attention-Based Multi-RNNs                                                                 | [DOI](https://doi.org/10.1109/LCOMM.2020.2995842) / [PDF](https://www.researchgate.net/publication/341533342_Radar_Emitter_Classification_With_Attention-Based_Multi-RNNs) | Paper method                                             | High           | Uses multiple RNN branches to process different pulse-stream features and fuses them with attention. Suitable for radar emitter classification under complex pulse-stream conditions                                                       |
| LSTM-based radar emitter type identification | Identification of Radar Emitter Type with Recurrent Neural Networks                                                          | [PDF](https://sspd.eng.ed.ac.uk/sites/sspd.eng.ed.ac.uk/files/attachments/basicpage/20200916/S2%20Apfeld.pdf) / [DOI](https://doi.org/10.1109/SSPD47486.2020.9271988)      | Paper method                                             | High           | Uses LSTM to model multifunction radar emission behavior sequences, focusing on resource-management and behavioral-grammar differences. Suitable for agile / multifunction radar type identification                                       |
| 1D CNN with attention                        | Radar Emitter Signal Recognition Based on One-Dimensional Convolutional Neural Network with Attention Mechanism              | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7664421/) / [MDPI](https://www.mdpi.com/1424-8220/20/21/6350)                                                             | Paper method                                             | Medium to High | Learns discriminative features directly from one-dimensional radar signal sequences and emphasizes key features using attention. More oriented toward intra-pulse signal recognition but can serve as a radar emitter recognition baseline |
| Knowledge-graph-driven CNN                   | A Knowledge Graph-Driven CNN for Radar Emitter Identification                                                                | [Paper](https://www.mdpi.com/2072-4292/15/13/3289) / [DOI](https://doi.org/10.3390/rs15133289)                                                                             | Paper method                                             | High           | Introduces knowledge graphs into radar emitter identification and uses expert knowledge and attribute relationships to support CNN feature selection and model design                                                                      |
| Joint multi-radar sorting and recognition    | A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and Time-Frequency Convolution Network | [Paper](https://www.sciencedirect.com/science/article/pii/S1051200425000272)                                                                                               | Paper method                                             | Medium to High | Jointly handles sorting and recognition in interleaved pulse streams. It first applies hierarchical clustering for multi-radar sorting and then uses a time-frequency convolutional network for emitter recognition                        |

---

### 4.2 Method Summaries

#### 4.2.1 Parameter-set Clustering and Classification

“Radar Emitter Recognition Based on Parameter Set Clustering and Classification” is a representative traditional machine-learning method closely related to radar emitter model identification. It takes PDW parameters and radar pulse parameter sets as inputs, first clusters parameter sets to extract stable parameter combinations and frequent items, and then feeds these structured features into a classifier for radar emitter recognition. Unlike methods that directly recognize modulation types, this method focuses on the overall behavioral characteristics of radar emitters at the parameter level. Therefore, it is more suitable for PDW / parameter-set-based radar model or emitter type identification.

#### 4.2.2 Agile-waveform Emitter Recognition

“Recognition of Radar Emitters with Agile Waveform Based on Hybrid Deep Neural Network and Attention Mechanism” addresses radar emitter recognition for agile-waveform radars with rapidly changing parameters. The paper defines the input as deinterleaved PDW pulse sequences, where each pulse contains PA, CF, PW, PRI, AOA, and other features. It jointly models the sequence using dynamic convolutional networks, LSTM, and attention. The convolutional network extracts local parameter-combination features, the LSTM learns temporal variation patterns in the pulse sequence, and the attention mechanism highlights key pulses and features. This is a representative method for multi-pulse PDW sequence-based radar emitter type recognition and has an associated open-source project, making it a priority candidate for reproduction.

#### 4.2.3 Transfer Learning and Online Learning

“Radar Emitter Identification under Transfer Learning and Online Learning” addresses limited target-domain samples, signal-distribution shifts, and dynamic model updates in radar emitter identification. The main value of this direction is to handle domain shifts commonly encountered in realistic electronic-intelligence environments, where training data and intercepted operational data may come from different environments, SNRs, or operating conditions. The method uses existing source-domain samples to support target-domain identification and updates the model gradually through online learning. This direction is highly suitable for practical radar emitter model identification systems.

#### 4.2.4 Attribute-specific RNN Sequence Modeling

“Radar Emitter Classification with Attribute-specific Recurrent Neural Networks” is a radar emitter classification method based on multivariate PDW sequences. It argues that different PDW attributes have different statistical properties and temporal variation patterns. Therefore, RF, PW, PRI, AOA, PA, and other features should not simply be concatenated and fed into a single model. Instead, attribute-specific recurrent neural-network branches should be designed to model different attributes separately. The model learns temporal patterns for each pulse-parameter attribute and then fuses the resulting sequence features to classify emitters. This method is suitable as a deep-learning baseline for PDW sequence-based radar emitter recognition.

#### 4.2.5 Attention-based Multi-RNN

“Radar Emitter Classification With Attention-Based Multi-RNNs” further develops multivariate sequence modeling by using multiple RNN branches to process different pulse-stream features and applying attention to adaptively weight the importance of different feature branches. This method is suitable for radar pulse streams in complex electromagnetic environments because different parameters may have different discriminative values under different noise conditions, work modes, or radar types. Attention helps the model emphasize more discriminative feature dimensions and temporal segments, making it a representative deep-fusion model for PDW sequence-based radar emitter classification.

#### 4.2.6 LSTM-based Radar Emitter Type Identification

“Identification of Radar Emitter Type with Recurrent Neural Networks” focuses on multifunction radar emitter type identification and uses LSTM to model radar emission behavior sequences. The paper does not focus on static classification of individual pulse parameters. Instead, it exploits the implicit “behavioral grammar” or resource-management pattern in pulse sequences to distinguish different radar types. For agile radars, multifunction radars, and cognitive radars, different radars may have similar parameter ranges, but their operating procedures, task-switching rules, and emission behavior sequences can differ. LSTM is suitable for capturing such long-term temporal dependencies and is therefore useful for radar emitter type identification and behavior-assisted recognition.

#### 4.2.7 1D CNN with Attention

“Radar Emitter Signal Recognition Based on One-Dimensional Convolutional Neural Network with Attention Mechanism” uses a one-dimensional convolutional neural network and attention mechanism to extract discriminative features directly from radar signal sequences. Unlike PDW-parameter-based methods, this class of methods learns local structural features from raw one-dimensional signals or intra-pulse sampled sequences, reducing the dependence on hand-crafted feature design. The attention mechanism emphasizes key segments and features in the signal sequence, allowing the model to focus on components that contribute most to emitter discrimination. This method is closer to radar emitter signal recognition than pure PDW-based model identification, but it can serve as an intra-pulse signal branch or an end-to-end recognition baseline.

#### 4.2.8 Knowledge-graph-driven CNN

“A Knowledge Graph-Driven CNN for Radar Emitter Identification” integrates radar-domain knowledge with deep-learning models by using a knowledge graph to support radar emitter identification. The core idea is to organize expert experience, parameter relationships, emitter attributes, and class relations into a knowledge graph, and then use this prior knowledge to guide feature selection, model-structure design, or recognition decisions. Compared with a fully end-to-end convolutional-network approach, a knowledge-graph-driven model emphasizes interpretability and prior knowledge utilization, making it closer to practical electronic-intelligence workflows that rely on expert rules, parameter libraries, and emitter knowledge bases. This direction is representative of knowledge-enhanced radar model identification and rule-deep-learning hybrid recognition.

#### 4.2.9 Joint Multi-radar Sorting and Recognition

“A Multi-Radar Emitter Sorting and Recognition Method Based on Hierarchical Clustering and Time-Frequency Convolution Network” jointly considers sorting and recognition in multi-radar interleaved pulse streams. It first applies hierarchical clustering to interleaved PDWs or pulse streams to separate pulses from different emitters as much as possible. Then, it uses time-frequency analysis and convolutional networks to identify the sorted signals. This method is closer to a realistic electronic-intelligence processing pipeline because actual received data are often not clean pulse sequences from a single radar, but mixed scenarios involving multiple emitters, pulse overlap, and complex parameter variations. Therefore, this direction is suitable as a representative method for “sorting-recognition joint modeling” or “radar model identification in complex multi-emitter scenarios”.

---

## 5. Recommended Experimental Settings

### 5.1 Suggested Baselines

| Category                     | Methods                                                                                                        | Usage                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Parameter-matching baselines | Parameter-template matching, expert rule libraries, nearest-neighbor template matching                         | Engineering-style model-identification baselines    |
| Traditional machine learning | SVM, random forest, gradient-boosted trees, k-nearest neighbors, Gaussian mixture models, hidden Markov models | PDW / parameter-feature classification              |
| Sequence modeling            | Hidden Markov models, LSTM, gated recurrent units, Transformer encoders                                        | PDW sequence and work-mode sequence recognition     |
| Time-frequency image methods | STFT / CWT / pseudo Wigner-Ville distribution + convolutional networks                                         | Intra-pulse structure-assisted recognition          |
| Raw-signal methods           | 1D CNN, temporal convolutional networks, Transformer                                                           | End-to-end IQ or one-dimensional signal recognition |
| Transfer learning            | TrAdaBoost, domain adaptation, online learning                                                                 | Few-shot target domains and online updates          |
| Open-set recognition         | OpenMax, energy score, prototype distance, metric autoencoder                                                  | Unknown emitter detection                           |
| Representation learning      | Autoencoder, contrastive learning, masked signal modeling                                                      | Pretraining and low-label identification            |

---

### 5.2 Suggested Pipelines

If using PDW / parameter-set inputs:

```text
PDW sequence / parameter set
        ↓
Feature cleaning and normalization
        ↓
Parameter statistics / temporal features / work-mode features
        ↓
Traditional classifier or deep sequence model
        ↓
Radar model / functional category / unknown-class decision
```

If using IQ or time-frequency image inputs:

```text
Raw IQ signal
        ↓
Preprocessing and normalization
        ↓
STFT / CWT / pseudo Wigner-Ville distribution or end-to-end encoder
        ↓
CNN / 1D CNN / Transformer
        ↓
Model class / waveform-assisted class / unknown-class decision
```

---

## 6. Notes

* The core task of this page is radar emitter model identification, not radar pulse deinterleaving, waveform recognition, modulation recognition, or specific emitter identification;
* Strict model identification requires labels corresponding to radar models, equipment families, functional categories, or known emitter classes;
* Public real radar model-level data are extremely limited, and related datasets must be clearly described as proxy tasks or auxiliary resources;
* Modulation classes, waveform classes, and hardware individual IDs should not be directly equated with radar models;
* When using a simulated model library, the RF, PRI, PW, modulation type, work modes, and parameter distributions of each model should be clearly specified;
* Reporting only accuracy is not recommended. Macro-F1, confusion matrices, cross-domain results, and open-set metrics should also be reported;
* Open-set recognition, cross-domain generalization, and online updates are closer to real electronic-intelligence scenarios than ordinary closed-set classification;
* When real radar signal data are involved, data sources, licenses, security compliance, and confidentiality requirements must be followed.

---

## 7. Lab Contributions and Patent Portfolio

In addition to public papers and open-source methods, our lab has conducted systematic research on radar emitter model identification and has developed several patent contributions for non-cooperative radar emitter recognition. These works mainly focus on contextual semantic modeling, multivariate time-series modeling, self-supervised representation learning, and joint intra-pulse / inter-pulse contrastive learning. They are complementary to existing PDW sequence identification, IQ-signal identification, and open-set recognition methods.

| No. | Patent / Method Name                                                                                                    | Country / Region | Direction                                            | Main Contribution                                                                                                                                                                                                              | Relationship to Radar Model Identification                                                                                                                           |
| --: | ----------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | A Contextual Semantic-aware Method for Radar Emitter Model Identification                                               | China            | Contextual semantic modeling                         | Mines contextual semantic relationships from radar pulse sequences or emitter behavior sequences, using dependencies among neighboring pulses, parameter variations, and operational behaviors to improve model identification | Suitable for multifunction or agile-radar scenarios where a single pulse is insufficient and sequence context is required for reliable identification                |
|   2 | A Multivariate Time-series Transformer-based Method for Radar Emitter Model Identification                              | China            | Multivariate time-series Transformer                 | Treats RF/CF, PW, PRI, PA, DOA/AOA, and other PDW parameters as multivariate time-series inputs, and uses Transformer to model long-range dependencies and cross-variable interactions                                         | Suitable for PDW sequence-based radar emitter model identification and can serve as an enhanced alternative to RNN / LSTM methods                                    |
|   3 | A Hybrid Self-supervised Learning-based Method for Non-cooperative Radar Emitter Model Identification                   | China            | Self-supervised representation learning              | Addresses the lack of labeled samples in non-cooperative scenarios by learning general representations from unlabeled radar signals or PDW data through self-supervised tasks and transferring them to model identification    | Suitable for real electronic-intelligence scenarios with scarce labels, large scenario shifts, and insufficient samples of new emitters                              |
|   4 | An Intra-pulse and Inter-pulse Contrastive Learning-based Method for Non-cooperative Radar Emitter Model Identification | China            | Joint intra-pulse / inter-pulse contrastive learning | Uses both intra-pulse signal features and inter-pulse PDW sequence features, and applies contrastive learning to make samples from the same emitter or model closer and samples from different emitters more separable         | Suitable for multimodal radar emitter model identification by fusing IQ / intermediate-frequency intra-pulse information with PDW inter-pulse behavioral information |

---

## 8. Citation and Contribution

This resource list is maintained by the **SmartDSP Lab, School of Informatics, Xiamen University**. It aims to provide a systematic, reproducible, and extensible open resource index for radar model identification and related intelligent electromagnetic-signal processing research.

Contributions through Issues or Pull Requests are welcome, including but not limited to:

* Newly released radar emitter model identification datasets;
* Public resources with labels clearly corresponding to radar models, functional categories, or emitter classes;
* Papers on radar emitter recognition, open-set recognition, cross-domain generalization, and online learning;
* Reproducible benchmark results for model identification;
* Clarifications on the boundaries among modulation recognition, waveform recognition, specific emitter identification, and model identification.

Recommended information for submitting a resource:

```text
Resource name:
Resource link:
Task type:
Label semantics:
Whether it strictly belongs to model identification:
Whether the data are open-source:
Whether the code is open-source:
Reason for recommendation:
Notes:
```

This repository is intended only for organizing public academic resources and supporting reproducible experimental references. It does not provide non-public radar databases, sensitive system parameters, or support for any unauthorized signal interception, identification, or electronic-countermeasure activities.
