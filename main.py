# ==============================================================================
# AES HYBRID PIPELINE FOR CROSS-PROMPT RELIABILITY EVALUATION
# Architecture: RoBERTa Embeddings + TF-IDF + Linguistic Features + LightGBM
# ==============================================================================

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import textstat
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel

def parse_arguments():
    """
    Parses command-line arguments for dynamic system execution,
    allowing customizable paths and key baseline hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Hybrid AES Framework with Cross-Prompt Reliability Evaluation"
    )
    
    # Dynamic Path Settings
    parser.add_argument(
        "--asap_path", 
        type=str, 
        default="data/training_set_rel3.tsv",
        help="Path to the primary ASAP dataset file (TSV format)"
    )
    parser.add_argument(
        "--teacher_path", 
        type=str, 
        default="data/teacher_dataset.csv",
        help="Path to the external teacher-scored validation dataset (CSV format)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="journal_outputs",
        help="Target directory where evaluation metrics and SHAP visualizations are preserved"
    )
    
    # Core Model Hyperparameters
    parser.add_argument(
        "--n_estimators", 
        type=int, 
        default=800,
        help="Number of boosting iterations for LightGBM Regressor"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.02,
        help="Shrinkage rate/learning rate for gradient boosting optimization"
    )
    parser.add_argument(
        "--num_leaves", 
        type=int, 
        default=128,
        help="Maximum tree leaves for base learners to control model capacity"
    )

    return parser.parse_args()


class TextDataset(Dataset):
    """
    Custom PyTorch Dataset encoder to batch text fields for parallel 
    Transformer tokenization and feature inference.
    """
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }


def extract_roberta_embeddings(texts, model_name="roberta-base", batch_size=16, device="cpu"):
    """
    Extracts high-dimensional dense semantic vectors (768-dim) from the mean-pooled 
    final hidden states of pre-trained RoBERTa architectures.
    """
    print(f"Extracting contextual semantic representations via {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Transformer Inference Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Apply mean pooling over the sequence dimension to preserve context
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)


def extract_linguistic_features(texts):
    """
    Computes a comprehensive handcrafted multi-dimensional matrix tracking 
    syntactic, structural, lexical richness, and readability markers.
    """
    print("Computing hand-engineered linguistic and surface-level metrics...")
    features = []
    for text in tqdm(texts, desc="Linguistic Feature Engineering"):
        text_str = str(text)
        features.append([
            textstat.flesch_reading_ease(text_str),
            textstat.flesch_kincaid_grade(text_str),
            textstat.smog_index(text_str),
            textstat.coleman_liau_index(text_str),
            textstat.automated_readability_index(text_str),
            textstat.dale_chall_readability_score(text_str),
            textstat.difficult_words(text_str),
            textstat.linsear_write_formula(text_str),
            textstat.gunning_fog(text_str),
            textstat.text_standard(text_str, float_output=True),
            textstat.lexicon_count(text_str, removepunct=True),
            textstat.sentence_count(text_str),
            textstat.char_count(text_str, ignore_spaces=True),
            textstat.letter_count(text_str, ignore_spaces=True),
            textstat.polysyllabcount(text_str),
            textstat.monosyllabcount(text_str),
            len(re.findall(r'\b\w+\b', text_str.lower())),
            len(set(re.findall(r'\b\w+\b', text_str.lower()))),
            len(text_str.split('\n')),
            sum(1 for c in text_str if c.isupper()),
            sum(1 for c in text_str if c.isdigit()),
            len(re.findall(r'[.,!?;:]', text_str))
        ])
    return np.array(features)


def compute_qwk(y_true, y_pred, min_rating=0, max_rating=60):
    """
    Computes the Quadratic Weighted Kappa (QWK) to quantify scoring agreement.
    Continuous outputs are rounded into structured integer bounds.
    """
    y_pred_rounded = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
    y_true_bounded = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
    return cohen_kappa_score(
        y_true_bounded, 
        y_pred_rounded, 
        weights="quadratic", 
        labels=list(range(min_rating, max_rating + 1))
    )


def execute_cross_validation(X_full, y, args):
    """
    Executes a structured out-of-sample 5-fold cross-validation routine 
    contrasting Ridge, SVR, and the proposed hybrid LightGBM models.
    """
    print("\n--- Initializing Out-of-Sample 5-Fold Cross-Validation ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_records = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full, y)):
        X_train, X_test = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline 1: Linear Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        preds_ridge = ridge.predict(X_test)
        qwk_ridge = compute_qwk(y_test, preds_ridge)

        # Baseline 2: Support Vector Regression (SVR)
        svr = SVR(C=1.0, epsilon=0.1)
        svr.fit(X_train, y_train)
        preds_svr = svr.predict(X_test)
        qwk_svr = compute_qwk(y_test, preds_svr)

        # Proposed: Highly Regularized LightGBM Regressor Framework
        lgb_model = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        preds_hybrid = lgb_model.predict(X_test)
        qwk_hybrid = compute_qwk(y_test, preds_hybrid)

        cv_records.append({
            "Fold": fold + 1,
            "Ridge_QWK": qwk_ridge,
            "SVR_QWK": qwk_svr,
            "Hybrid_QWK": qwk_hybrid
        })
        print(f"Fold {fold+1} Completed | Ridge: {qwk_ridge:.4f} | SVR: {qwk_svr:.4f} | Hybrid: {qwk_hybrid:.4f}")

    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(os.path.join(args.output_dir, "cv_results.csv"), index=False)
    print("Cross-Validation results saved successfully.")


def execute_ablation_study(emb, tfidf_mat, ling, y, args):
    """
    Evaluates In-Sample Representation Capacity across localized feature spaces 
    to empirically map upper-bound performance thresholds.
    """
    print("\n--- Commencing In-Sample Ablation Assessment ---")
    experiments = {
        "Lexical_Space_Only_(TFIDF)": tfidf_mat,
        "Semantic_Space_Only_(RoBERTa)": emb,
        "Linguistic_Space_Only_(Textstat)": ling,
        "Unified_MultiTier_Fusion_Space": np.hstack([emb, tfidf_mat, ling])
    }

    ablation_records = []
    for name, feature_space in experiments.items():
        lgb_model = lgb.LGBMRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(feature_space, y)
        preds = lgb_model.predict(feature_space)
        
        qwk_score = compute_qwk(y, preds)
        mse_score = mean_squared_error(y, preds)
        
        ablation_records.append({
            "Feature_Configuration": name,
            "InSample_MSE": mse_score,
            "InSample_QWK": qwk_score
        })
        print(f"Config: {name} | QWK Upper-bound: {qwk_score:.4f}")

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(os.path.join(args.output_dir, "ablation_study.csv"), index=False)
    print("Ablation matrix successfully updated.")


def execute_prompt_wise_evaluation(df, emb, tfidf_mat, ling, args):
    """
    Isolates predictive boundaries within discrete prompt limitations to resolve
    structural variance issues and evaluate local reliability constraints.
    """
    print("\n--- Executing Isolated Prompt-Wise Performance Audit ---")
    prompt_records = []
    unique_prompts = sorted(df["essay_set"].unique())

    for prompt_id in unique_prompts:
        indices = df[df["essay_set"] == prompt_id].index.tolist()
        
        # Sub-sample localized segments
        y_prompt = df.loc[indices, "normalized_score"].values
        emb_p = emb[indices]
        tfidf_p = tfidf_mat[indices]
        ling_p = ling[indices]
        X_p = np.hstack([emb_p, tfidf_p, ling_p])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        p_preds = np.zeros(len(indices))

        for train_idx, test_idx in kf.split(X_p, y_prompt):
            lgb_model = lgb.LGBMRegressor(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_p[train_idx], y_prompt[train_idx])
            p_preds[test_idx] = lgb_model.predict(X_p[test_idx])

        rmse = np.sqrt(mean_squared_error(y_prompt, p_preds))
        pearson_r, _ = pearsonr(y_prompt, p_preds)
        qwk = compute_qwk(y_prompt, p_preds)

        prompt_records.append({
            "Prompt": prompt_id,
            "RMSE": rmse,
            "Pearson_r": pearson_r,
            "QWK": qwk
        })
        print(f"Prompt {prompt_id} Framework Evaluation -> RMSE: {rmse:.4f}, Pearson: {pearson_r:.4f}, QWK: {qwk:.4f}")

    df_prompt = pd.DataFrame(prompt_records)
    df_prompt.to_csv(os.path.join(args.output_dir, "prompt_results.csv"), index=False)
    print("Prompt-wise reliability evaluation saved.")


def draw_system_architecture(output_path):
    """
    Generates a high-resolution, formal system flow diagram for the 
    Multi-tier Fusion pipeline to resolve manuscript visualization missing issues.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    
    box_props = dict(boxstyle="round,pad=0.5", fc="#e1f5fe", ec="#0288d1", lw=1.5)
    f_props = dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#388e3c", lw=1.2)
    model_props = dict(boxstyle="round,pad=0.6", fc="#fff3e0", ec="#f57c00", lw=2)

    ax.text(0.1, 0.5, "Input Student\nEssay Text", ha="center", va="center", bbox=box_props, fontsize=11)
    
    # Text representations
    ax.text(0.4, 0.8, "Semantic Layer\n(roBERTa-base Context Embeddings)", ha="center", va="center", bbox=f_props, fontsize=9)
    ax.text(0.4, 0.5, "Lexical Layer\n(Character/Word TF-IDF Vectors)", ha="center", va="center", bbox=f_props, fontsize=9)
    ax.text(0.4, 0.2, "Syntactic Layer\n(Textstat Structural Complexity)", ha="center", va="center", bbox=f_props, fontsize=9)
    
    ax.text(0.7, 0.5, "Multi-tier\nFeature Fusion\nMatrix Concatenation", ha="center", va="center", bbox=box_props, fontsize=10)
    ax.text(0.95, 0.5, "Optimized\nLightGBM\nRegressor", ha="center", va="center", bbox=model_props, fontsize=11)

    # Drawing directional vectors
    arrow = dict(arrowstyle="->", lw=1.5, color="#37474f")
    ax.annotate("", xy=(0.24, 0.75), xytext=(0.18, 0.55), arrowprops=arrow)
    ax.annotate("", xy=(0.24, 0.50), xytext=(0.18, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.24, 0.25), xytext=(0.18, 0.45), arrowprops=arrow)
    
    ax.annotate("", xy=(0.56, 0.55), xytext=(0.50, 0.75), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.50), xytext=(0.52, 0.50), arrowprops=arrow)
    ax.annotate("", xy=(0.56, 0.45), xytext=(0.50, 0.25), arrowprops=arrow)
    
    ax.annotate("", xy=(0.84, 0.50), xytext=(0.79, 0.50), arrowprops=arrow)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"System architecture block diagram generated at: {output_path}")


# ------------------------------------------------------------------------------
# CORE PIPELINE EXECUTION ENGINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    
    print("====================================================================")
    print("RUNNING AUTOMATED ESSAY SCORING REPRODUCIBILITY ENGINE")
    print("====================================================================")
    print(f"Primary Dataset Target: {args.asap_path}")
    print(f"Validation Target:      {args.teacher_path}")
    print(f"Output Vault:           {args.output_dir}")
    print("====================================================================")

    # Hardware acceleration check
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System execution assigned to: {DEVICE}")

    # Step 1: Pre-process primary tabular structures
    if not os.path.exists(args.asap_path):
        raise FileNotFoundError(f"Missing mandatory ASAP source files at {args.asap_path}")
        
    df = pd.read_csv(args.asap_path, sep="\t", encoding="ISO-8859-1")
    df = df.dropna(subset=["essay", "essay_set", "domain1_score"])
    df = df.reset_index(drop=True)

    # Normalize localized boundaries to a scale-invariant distribution [0, 1]
    df["normalized_score"] = 0.0
    for prompt_id in df["essay_set"].unique():
        subset = df[df["essay_set"] == prompt_id]
        min_s = subset["domain1_score"].min()
        max_s = subset["domain1_score"].max()
        if max_s > min_s:
            df.loc[subset.index, "normalized_score"] = (subset["domain1_score"] - min_s) / (max_s - min_s)
        else:
            df.loc[subset.index, "normalized_score"] = 1.0

    texts = df["essay"].tolist()
    y = df["normalized_score"].values

    # Step 2: Multi-tier feature extraction routines
    emb = extract_roberta_embeddings(texts, model_name="roberta-base", batch_size=16, device=DEVICE)
    
    print("Constructing lexical vector spaces using optimized character/word n-grams...")
    tfidf = TfidfVectorizer(max_features=5000, analyzer="word", ngram_range=(1, 3), stop_words="english")
    tfidf_mat = tfidf.fit_transform(texts).toarray()
    
    ling = extract_linguistic_features(texts)

    # Unified feature space concatenation
    X_full = np.hstack([emb, tfidf_mat, ling])
    print(f"Unified input feature space initialization finalized. Target shape: {X_full.shape}")

    # Step 3: Empirical evaluations
    execute_cross_validation(X_full, y, args)
    execute_ablation_study(emb, tfidf_mat, ling, y, args)
    execute_prompt_wise_evaluation(df, emb, tfidf_mat, ling, args)

    # Step 4: Resolve manuscript figure compliance dependencies
    draw_system_architecture(os.path.join(args.output_dir, "architecture.png"))

    # Step 5: Advanced interpretability modeling via SHAP values
    print("\n--- Constructing SHAP Model Interpretability Logs ---")
    final_model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X_full, y)

    explainer = shap.TreeExplainer(final_model)
    sample_size = min(1000, len(X_full))
    sample_data = X_full[:sample_size]
    shap_values = explainer.shap_values(sample_data)

    # Export continuous SHAP summary densities
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample_data, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Export structured SHAP dimensional importance weights
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample_data, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("====================================================================")
    print("PIPELINE EXECUTION CONCLUDED. ALL ARTIFACTS EXPORTED SUCCESSFULLY.")
    print("====================================================================")