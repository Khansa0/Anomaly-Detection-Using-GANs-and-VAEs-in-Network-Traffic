# Anomaly Detection Using GANs and Variational Autoencoders in Network Traffic

Research project investigating generative deep-learning methods for handling class imbalance and detecting malicious network traffic using the **CICIDS2017 intrusion-detection dataset**. The main pipeline combines **Generative Adversarial Networks (GANs)**, a **β-Variational Autoencoder (β-VAE)**, and **LightGBM**.

> **Results note:** The metrics below are historical results reported for the thesis experiment. They should be treated as results of that specific experiment and not as results from any subsequently revised implementation unless that implementation has been rerun and verified.

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objectives](#research-objectives)
- [Main Pipeline](#main-pipeline)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Components](#model-components)
- [Reported Results](#reported-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations and Validation](#visualizations-and-validation)
- [Repository Structure](#repository-structure)
- [Environment and Setup](#environment-and-setup)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)
- [Author](#author)

## Project Overview

Network intrusion detection is challenging because benign traffic and different attack categories are not evenly represented. A model may perform well on frequent classes while missing less common attacks. This project studies whether GAN-based synthetic data generation, followed by β-VAE representation learning and LightGBM classification, can improve multiclass network intrusion detection under class imbalance.

The research evaluates the complete **GAN + β-VAE + LightGBM** pipeline. The project is intended for academic research and experimentation; results depend on preprocessing, split strategy, model settings, and the version of the code used.

## Research Objectives

- Investigate GAN-based generation of synthetic attack-traffic samples.
- Study the use of synthetic data to address class imbalance in network traffic.
- Learn latent feature representations using a β-VAE.
- Use LightGBM for downstream multiclass traffic classification.
- Evaluate performance using accuracy, precision, recall, F1-scores, and class-wise analysis.
- Examine the strengths and limitations of generative augmentation for minority attack detection.

## Main Pipeline

```text
CICIDS2017 CSV files
        |
        v
Data cleaning and preprocessing
        |
        v
Feature scaling and label preparation
        |
        v
GAN-based synthetic attack-data generation
        |
        v
Combine real training data and synthetic samples
        |
        v
β-VAE latent representation learning
        |
        v
LightGBM classifier
        |
        v
Held-out evaluation and class-wise metrics
```

The exact architecture, class-balancing target, split strategy, and hyperparameters should be taken from the code version used to generate a particular set of results.

## Dataset

**CICIDS2017 — Canadian Institute for Cybersecurity**

CICIDS2017 contains benign traffic and multiple attack categories and is commonly used for network intrusion-detection research.

The dataset is not included in this repository. Download it from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html), review its terms, and configure the local paths expected by the scripts.

## Preprocessing

The experiment's preprocessing workflow includes operations such as:

- Combining the required CSV files.
- Removing identifier or non-predictive columns where appropriate (for example, flow identifiers, IP addresses, ports, timestamps, and unnamed index columns).
- Handling missing and invalid values.
- Preparing class labels.
- Scaling numerical features.
- Splitting data for model development and evaluation.

**Leakage prevention:** For a rigorous evaluation, fit preprocessing transformations such as the scaler on training data only, then apply the fitted transformation to validation and test data. Keep the test set separate from model selection and early stopping. Check the implementation in the exact experiment version before claiming these safeguards were applied.

## Model Components

### 1. Generative Adversarial Network (GAN)

The GAN learns to generate synthetic feature vectors resembling real training samples. In this project, GAN-based augmentation is intended to increase the representation of underrepresented attack classes. Synthetic samples should be evaluated for distributional similarity and potential memorization, rather than assumed to be valid solely because the GAN trains successfully.

### 2. β-Variational Autoencoder (β-VAE)

The β-VAE learns a lower-dimensional latent representation. Its objective combines reconstruction loss and a weighted Kullback–Leibler divergence term:

\[
\mathcal{L}_{\beta\text{-VAE}} =
\mathcal{L}_{\mathrm{reconstruction}} +
\beta D_{\mathrm{KL}}\left(q(z|x)\,\|\,p(z)\right)
\]

The value of β controls the weight assigned to latent-distribution regularization. Its effect on classification performance should be established through experiments.

### 3. LightGBM

LightGBM is used as the downstream classifier on the learned feature representation. The evaluation should include class-wise metrics because overall accuracy can conceal weak detection of rare attacks.

## Reported Results

The following metrics were reported for the thesis's **GAN + β-VAE + LightGBM** experiment on CICIDS2017.

| Metric | Reported result |
|---|---:|
| Accuracy | **98.63%** |
| Weighted F1-score | **98.71%** |
| Macro-F1 | **75.48%** |
| Minority-class F1 (average over selected minority classes) | **64.94%** |

### Interpretation

- **Accuracy (98.63%)** measures the overall proportion of correctly classified records.
- **Weighted F1 (98.71%)** accounts for class support, so frequent classes contribute more to the aggregate.
- **Macro-F1 (75.48%)** gives each class equal weight and provides a more informative view when class frequencies differ.
- **Minority-class F1 (64.94%)** summarizes F1 across the selected minority classes; the exact class list and averaging method should be documented alongside the experiment output.

The difference between weighted F1 and macro-F1 indicates that performance should not be discussed using accuracy alone. Include the per-class classification report and confusion matrix in the repository when available.

> **Reproducibility warning:** These are previously reported experiment results, not independently verified results from the revised code. If the code, data split, preprocessing, or training protocol changes, rerun the experiment and update this table only with the newly generated metrics. Do not claim a result was reproduced unless the run and its outputs are available.

## Evaluation Metrics

| Metric | What it indicates |
|---|---|
| Accuracy | Overall fraction of correct predictions |
| Precision | How many predicted members of a class are correct |
| Recall | How many actual members of a class are detected |
| Weighted F1-score | F1 averaged with class support as weights |
| Macro-F1 | Unweighted mean of class-level F1-scores |
| Minority-class F1 | F1 averaged across the explicitly selected minority classes |
| Confusion matrix | Correct and incorrect predictions for each class |

## Visualizations and Validation

Add the actual generated figures and reports to a results folder and link them here once available. Useful artifacts include:

- Class distribution before and after augmentation.
- GAN training diagnostics.
- PCA and t-SNE plots, with captions explaining what they show.
- Confusion matrix and per-class classification report.
- Synthetic-versus-real feature-distribution comparisons.
- Statistical synthetic-data quality metrics and nearest-neighbour diagnostics.
- Experiment configuration, logs, and saved metric tables.

Loss curves alone do not prove GAN convergence or synthetic-data quality. PCA and t-SNE are exploratory visualizations, not standalone evidence of class separability or generalization.

## Repository Structure

The current repository view shows the following top-level items:

```text
Anomaly-Detection-Using-GANs-and-VAEs-in-Network-Traffic/
├── CODES/
├── Dataset/
└── README.md
```

Recommended additions, as the project is organized further:

```text
├── results/
│   ├── metrics/
│   ├── figures/
│   └── logs/
├── docs/
│   └── methodology.md
├── requirements.txt
└── .gitignore
```

Update this structure to reflect the files actually committed to the repository. Avoid uploading the raw CICIDS2017 dataset unless you have confirmed that redistribution is permitted; usually it is better to provide download instructions and keep the data local.

## Environment and Setup

The project was developed with Python and machine-learning libraries including TensorFlow/Keras, PyTorch, LightGBM, NumPy, pandas, scikit-learn, and Matplotlib. The exact required versions should be recorded in a tested `requirements.txt`.

1. Install a compatible Python version and Git.
2. Clone this repository:

   ```bash
   git clone https://github.com/Khansa0/Anomaly-Detection-Using-GANs-and-VAEs-in-Network-Traffic.git
   cd Anomaly-Detection-Using-GANs-and-VAEs-in-Network-Traffic
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the dependencies listed in `requirements.txt` once that file has been generated and tested:

   ```bash
   pip install -r requirements.txt
   ```

5. Download CICIDS2017 from its official source, place the CSV files in the location expected by the scripts, and update local paths in the configuration or code.
6. Run the relevant experiment script from the `CODES/` folder. The exact command depends on the script currently in the repository; document the tested entry point here.

The setup commands above describe the environment workflow; they do not guarantee that all scripts will run without adjustment until the dependency file and execution instructions are verified.

## Reproducibility Notes

For a reproducible report, record:

- The exact script version and configuration used.
- Python and library versions.
- Random seeds and whether results are averaged over independent runs.
- Train/validation/test split strategy.
- Preprocessing steps and fitted transformation parameters.
- GAN and β-VAE architecture and hyperparameters.
- Synthetic sample counts per class.
- Class-weighting strategy.
- Per-class metrics, confusion matrix, and saved logs.

A random split and a chronological or session-aware split measure different generalization settings. State clearly which protocol was used. Report mean and standard deviation across multiple runs only if those runs were actually completed.

## Limitations and Future Work

- Assess whether synthetic samples improve minority-class detection without introducing memorization or unrealistic patterns.
- Compare the complete pipeline against suitable baselines using an identical split and evaluation protocol.
- Investigate augmentation-ratio sensitivity.
- Evaluate temporal, session-aware, or source-grouped splits where the dataset structure supports them.
- Repeat experiments with multiple seeds and report variability.
- Explore conditional generative models and alternative approaches to imbalanced intrusion detection.
- For external datasets, distinguish independent evaluation after retraining from true cross-dataset transfer, where a model trained on one dataset is tested on another without retraining.

## Reference

Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. Proceedings of the International Conference on Information Systems Security and Privacy (ICISSP).

Add complete bibliographic details for the GAN, β-VAE, LightGBM, and other methods cited in the thesis.

## Author

**Khansa Shahwar**  
MS Computer Science  
Research interests: generative AI, cybersecurity, network intrusion detection, anomaly detection, GANs, VAEs, and machine learning.

## Disclaimer

This repository is for academic and research purposes. Reported metrics apply to the stated experiment and should not be interpreted as a guarantee of performance on other datasets or in real-world networks.

## License

Add a license only after deciding how the code and documentation may be reused. Review the rights and terms applying to third-party code, datasets, and research material before publishing.
