# 🧬 Bio-Seq LM — DNA & Protein Language Model Pipelines

<div align="center">

**Design and Evaluation of DNA & Protein Language Model Pipelines for Biological Sequence Analysis**

*M.S. Applied Data Science Capstone · University of Florida · Spring 2026*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HiPerGator](https://img.shields.io/badge/HPC-HiPerGator_A100-0021A5?style=flat-square)](https://www.rc.ufl.edu/about/hipergator/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[🚀 Live App](#-interactive-app--bio-seq-lm-explorer) · [📊 Results](#-results) · [📓 Notebooks](#-notebooks) · [🛠 Setup](#-setup)

</div>

---

## 📌 Overview

Do large pretrained biological language models **consistently outperform engineered baselines** — or is their advantage task-dependent?

This project answers that question through a rigorous, end-to-end comparison of **5 modeling paradigms** across **2 biological sequence tasks**:

| Task | Data | Scale | Primary Metric |
|------|------|-------|----------------|
| 🧬 **DNA Promoter Classification** | Human GRCh38 · 200 bp sequences | 4,000 sequences (balanced) | ROC-AUC |
| 🔬 **Protein Pfam Classification** | UniProt Swiss-Prot · 10 Pfam families | 2,293 sequences | Accuracy + F1-macro |

---

## 🏆 Results

### DNA — Promoter vs Non-Promoter Classification

| Paradigm | Best Model | Accuracy | ROC-AUC |
|----------|-----------|----------|---------|
| Baseline ML | Random Forest | 0.748 | 0.837 |
| CNN | Seq-CNN | 0.750 | 0.826 |
| Hybrid | CNN + features | 0.698 | 0.832 |
| DNABERT-2 | XGBoost | 0.756 | 0.846 |
| **NT-500M ★** | **XGBoost** | **0.769** | **0.851** |

> **Key finding:** NT-500M/XGBoost achieves ROC-AUC 0.851 — only **+1.5 pts above Random Forest baseline**. Engineered CpG/GC features remain highly competitive for this task.

### Protein — Pfam Family Classification

| Paradigm | Best Model | Accuracy | F1-macro |
|----------|-----------|----------|---------|
| Baseline ML | Random Forest | 0.923 | 0.924 |
| CNN | Seq-CNN | 0.924 | 0.925 |
| Hybrid | CNN + features | 0.956 | 0.957 |
| ESM-2 | Linear SVM | 0.987 | 0.989 |
| **ProtBERT ★** | **Linear SVM** | **0.991** | **0.992** |

> **Key finding:** ProtBERT/Linear SVM achieves **99.1% accuracy — +7 pts over Hybrid**. Pfam families are linearly separable in UniRef100 pretrained embedding space.

---

## 🖥️ Interactive App — Bio-Seq LM Explorer

A full-featured Streamlit application for real-time biological sequence analysis — no retraining required.

> 🎬 **[Video Demo](https://drive.google.com/drive/folders/1_e-JD7NuzYQWnCyiY3f02vL_0UVEKG72)** *(link coming soon)*

### App Features

| Tab | Description |
|-----|-------------|
| 🧬 **DNA Classification** | Classify any 200 bp sequence across 5 paradigms simultaneously. Includes in-silico mutagenesis and embedding space visualization |
| 🔬 **Protein Classification** | Pfam family prediction across 5 paradigms with alanine scanning and top-3 family confidence |
| 🔍 **Sequence Similarity** | Cosine similarity search in transformer embedding space (DNABERT-2, NT-500M, ESM-2, ProtBERT) |
| 🌌 **Embedding Explorer** | Interactive PCA of 4,000 DNA and 2,293 protein sequences — hover to inspect |
| ✂️ **Sequence Clustering** | Unsupervised KMeans clustering of 2–10 sequences in feature space with pairwise similarity heatmap |
| 📊 **Results Dashboard** | Full paradigm comparison with charts and key findings narrative |

### App Screenshots

<table>
<tr>
<td width="50%">

**DNA Classification — Multi-paradigm predictions**
![DNA Classification](docs/screenshots/app_dna_classification.png)

</td>
<td width="50%">

**Protein Classification — Pfam family prediction**
![Protein Classification](docs/screenshots/app_protein_classification.png)

</td>
</tr>
<tr>
<td width="50%">

**In-Silico Mutagenesis — Per-nucleotide sensitivity**
![Mutagenesis](docs/screenshots/app_mutagenesis.png)

</td>
<td width="50%">

**Embedding Explorer — ESM-2 & ProtBERT PCA**
![Embedding Explorer](docs/screenshots/app_embedding_explorer.png)

</td>
</tr>
<tr>
<td width="50%">

**Sequence Similarity Search — Nearest neighbours in LM space**
![Similarity Search](docs/screenshots/app_similarity_search.png)

</td>
<td width="50%">

**Results Dashboard — Full paradigm comparison**
![Results Dashboard](docs/screenshots/app_results_dashboard.png)

</td>
</tr>
</table>

---
## 🔬 Methodology

```
Raw Sequences (GRCh38 DNA · UniProt Protein)
            │
    ┌───────┴────────┐
    ▼                ▼
Feature Engineering  Transformer Embeddings
95 DNA features      DNABERT-2 · NT-500M (DNA)
176 Protein features ESM-2 · ProtBERT (Protein)
    │                │
    └───────┬────────┘
            ▼
  Downstream Classifiers
  LR · SVM · RF · XGBoost
            │
            ▼
  Evaluation & Comparison
  ROC-AUC (DNA) | Accuracy + F1 (Protein)
```

### Modeling Paradigms

1. **Baseline ML** — Classical models (LR, SVM, RF, GBM, XGBoost) on engineered biological features
2. **Sequence CNN** — 1D convolutional networks on one-hot encoded sequences; learns motifs automatically
3. **Hybrid** — CNN sequence embeddings concatenated with engineered features
4. **DNA LMs** — DNABERT-2 (117M params, BPE, 135-species) and NT-500M (500M params, GRCh38)
5. **Protein LMs** — ESM-2 (35M params, 65M UR50D sequences) and ProtBERT (420M params, 217M UniRef100)

### Feature Engineering

**DNA (95 features):** GC content, CpG O/E ratio, k-mer frequencies (di/tri-nucleotide), Shannon entropy, homopolymer runs, dinucleotide composition

**Protein (176 features):** CTD descriptors, PseAAC, reduced alphabet dipeptides, amino acid composition, physicochemical properties (GRAVY, instability, aromaticity)

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00 | `00_project_overview` | Project design, experiment plan, research question |
| 01 | `01_protein_ingest` | UniProt Swiss-Prot data loading and curation |
| 02 | `02_protein_eda` | Amino acid composition, length distributions, class balance |
| 03 | `03_dna_ingest` | GRCh38 promoter/non-promoter extraction |
| 04 | `04_dna_eda` | GC content, CpG analysis, sequence statistics |
| 05 | `05_dna_baseline_features` | DNA feature engineering (95 features) |
| 06 | `06_dna_baseline_models` | Classical ML baselines — RF achieves AUC 0.837 |
| 07 | `07_dna_sequence_models` | 1D CNN on one-hot DNA — AUC 0.826 |
| 08 | `08_dna_hybrid_model` | CNN + feature hybrid — AUC 0.832 |
| 09 | `09_protein_baseline_features` | Protein feature engineering (176 features) |
| 10 | `10_protein_baseline_models` | Classical ML baselines — RF accuracy 0.923 |
| 11 | `11_protein_sequence_models` | CNN on amino acid sequences — accuracy 0.924 |
| 12 | `12_protein_hybrid_model` | Hybrid CNN + features — accuracy 0.956 |
| 13 | `13_protein_esm2_embeddings` | ESM-2 embedding extraction |
| 14 | `14_protein_esm2_models` | Classifiers on ESM-2 — accuracy 0.987 |
| 15 | `15_dna_dnabert2_embeddings` | DNABERT-2 embedding extraction on HiPerGator A100 |
| 16 | `16_dna_dnabert2_models` | Classifiers on DNABERT-2 embeddings — AUC 0.846 |
| 17 | `17_dna_nt_embeddings` | NT-500M embedding extraction |
| 18 | `18_dna_nt_models` | Classifiers on NT-500M — AUC 0.851 ★ |
| 19 | `19_protein_protbert_embeddings` | ProtBERT embedding extraction |
| 20 | `20_protein_protbert_models` | Classifiers on ProtBERT — accuracy 0.991 ★ |
| 21 | `21_final_comparison` | Full cross-paradigm comparison and visualizations |

---

## 🗂️ Repository Structure

```
bio-ai-capstone/
│
├── app/
│   └── app-new.py               # Streamlit app — Bio-Seq LM Explorer
│
├── configs/
│   └── config.yaml              # Experiment configuration
│
├── data/
│   ├── raw/                     # Original downloaded datasets
│   ├── interim/                 # Intermediate processing files
│   └── processed/               # Clean datasets used for modeling
│       ├── dna_promoter_vs_nonpromoter_len200_pos2000_neg2000.csv
│       ├── protein_uniprot_pfam_top10_per400.csv
│       └── protein_esm2_embeddings_top10_per400.npy
│
├── docs/
│   ├── screenshots/             # Streamlit app screenshots
│   ├── poster/                  # Capstone poster (PDF)
│   └── presentation/            # Final presentation slides
│
├── models/
│   ├── dna/                     # Trained DNA models
│   │   ├── baselines/
│   │   ├── sequence_cnn/
│   │   └── hybrid/
│   └── protein/                 # Trained protein models
│       ├── baselines/
│       ├── sequence_cnn/
│       ├── hybrid_cnn_feats/
│       └── esm2/
│
├── notebooks/                   # NB00–NB21 full experiment pipeline
├── reports/                     # CSVs, JSONs with all experimental results
├── scripts/                     # Utility scripts
├── src/                         # Reusable source code
│   ├── common/
│   ├── dna/
│   └── protein/
│
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
└── README.md
```

---

## 🛠 Setup

### Prerequisites

- Python 3.11
- CUDA-capable GPU (A100 recommended for transformer embedding extraction)
- conda

### Installation

```bash
# Clone the repository
git clone https://github.com/deepikapratapa/bio-ai-capstone.git
cd bio-ai-capstone

# Create conda environment
conda env create -f environment.yml
conda activate bio-seq-lm

# Or pip install
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
cd app
streamlit run app-new.py
```

> **Note:** Transformer embeddings are pre-extracted and cached as `.npy` arrays. The app runs inference using pre-trained classifiers — no GPU required for the app itself.

### Running Notebooks

Notebooks are designed to run on **UF HiPerGator** with SLURM + NVIDIA A100 GPUs. For local runs, a CUDA-capable GPU is needed for notebooks 15–20 (transformer embedding extraction). Notebooks 00–14 run on CPU.

```bash
# On HiPerGator
cd /home/<username>/Capstone
jupyter lab --no-browser --port=8888
```

---

## 🔑 Key Takeaways

1. **Transformer models dramatically improve protein classification** — ProtBERT achieves +7 pp accuracy over the hybrid model. Pfam families are already linearly separable in UniRef100 pretrained embedding space.

2. **DNA classification is driven by biological features** — GC content and CpG density remain the most discriminative signals. NT-500M adds only +1.5 pp ROC-AUC over Random Forest.

3. **Protein embedding spaces are highly separable** — PCA of ESM-2 and ProtBERT embeddings shows tight, well-separated clusters for all 10 Pfam families.

4. **Interpretability reveals biologically meaningful signals** — In-silico mutagenesis identifies position-specific nucleotide sensitivity; alanine scanning pinpoints critical residues for family identity.

5. **The unified pipeline generalizes** — The app classifies any unseen DNA or protein sequence across all paradigms without retraining.

---

## 📊 Evaluation Metrics

| Metric | Task | Purpose |
|--------|------|---------|
| ROC-AUC | DNA | Primary metric; classification separability under class balance |
| PR-AUC | DNA | Performance robustness |
| Accuracy | Protein | Primary metric; 10-class balanced evaluation |
| F1-macro | Protein | Balanced per-class performance |
| 5-fold CV | Both | Robust generalization estimate |

---

## 🖥️ Infrastructure

| Component | Details |
|-----------|---------|
| HPC Cluster | UF HiPerGator |
| GPU | NVIDIA A100 (40 GB VRAM) |
| Scheduler | SLURM |
| Environment | Python 3.11 · PyTorch 2.1 · HuggingFace Transformers 4.1 · scikit-learn 1.3 · Streamlit 1.32 |
| Reproducibility | All random seeds fixed (seed=42); transformer embeddings pre-extracted and cached |

---

## 📚 References

1. Ji et al. (2023). DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. *arXiv:2306.15006*
2. Dalla-Torre et al. (2023). The Nucleotide Transformer. *bioRxiv*
3. Elnaggar et al. (2021). ProtTrans: Cracking the Language of Life Code Through Self-Supervised Learning. *IEEE TPAMI*
4. Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science 379(6637)*
5. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR 12:2825–2830*
6. Wolf et al. (2020). HuggingFace's Transformers: State-of-the-art NLP. *arXiv:1910.03771*

---

## 👥 Team

| Role | Name | Contact |
|------|------|---------|
| **Author** | Deepika Sarala Pratapa | [dpratapa@ufl.edu](mailto:dpratapa@ufl.edu) |
| **Faculty Advisor** | Matt Gitzendanner | [magitz@ufl.edu](mailto:magitz@ufl.edu) |
| **Instructor** | Edwin Marte Zorrilla | [emartezorrilla@ufl.edu](mailto:emartezorrilla@ufl.edu) |

**M.S. Applied Data Science · University of Florida · 2026**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/deepika-sarala-pratapa-3b8a29232)
[![GitHub](https://img.shields.io/badge/GitHub-deepikapratapa-181717?style=flat-square&logo=github)](https://github.com/deepikapratapa)

---

<div align="center">
<sub>Trained and deployed on UF HiPerGator HPC · NVIDIA A100 GPUs · SLURM scheduler · 40 GB VRAM</sub>
</div>
