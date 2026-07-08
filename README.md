# Automated Essay Scoring (AES) Reliability Across Essay Prompts

This repository contains the official implementation, experimental configurations, and interpretability pipeline for the manuscript: "Automated Essay Scoring Reliability Across Essay Prompts: An Empirical Study in Educational Assessment".

The framework introduces a Multi-tier Feature Fusion network engineered to analyze the structural stability and generalization boundaries of automated scoring models across diverse, prompt-isolated evaluation spaces.

---

## Framework Architecture and Key Components

The system integrates three distinct linguistic representation tiers to construct a comprehensive scoring space:
1. Semantic Embeddings: High-dimensional contextual vectors extracted via the contextualized CLS token representations of a pre-trained roBERTa-base encoder (768 dimensions).
2. Lexical Features: Sparse token frequency distributions captured via a TF-IDF Vectorizer (capped at 5,000 dimensions).
3. Syntactic and Surface Indicators: Core structural and readability metrics computed via the textstat engine (including length statistics, word complexity, and standardized readability indices like Flesch Reading Ease).

The combined feature matrix (exactly 5,790 features) is optimized and mapped to essay scores utilizing a regularized LightGBM Regressor.

---

## Repository Structure

```text
├── data/
│   └── .gitkeep                 # Local directory for raw benchmark datasets
├── journal_outputs/
│   └── .gitkeep                 # Target folder for training logs and metric tables
├── .gitignore                   # Safe configuration to exclude heavy files and caches
├── main.py                      # Main executable for feature extraction and model evaluation
├── README.md                    # Documentation file
└── requirements.txt             # Comprehensive Python environment dependencies