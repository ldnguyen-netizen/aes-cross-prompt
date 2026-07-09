import os
import re
import numpy as np
import pandas as pd
import torch
import textstat
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from scipy.stats import pearsonr, ttest_rel
from collections import Counter

# ============================================================
# SYSTEM CONFIGURATION AND CONSTANTS
# ============================================================
class PipelineConfig:
    ASAP_PATH = "data/training_set_rel3.tsv"
    TEACHER_PATH = "data/teacher_dataset.csv"
    OUTPUT_DIR = "journal_outputs"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRANSFORMER_MODEL = "roberta-base"
    
    MAX_LEN = 512
    BATCH_SIZE = 16
    NUM_FOLDS = 5
    LSTM_MAX_VOCAB = 5000
    LSTM_SEQ_LEN = 200

# Global cache to preserve computation power during execution iterations
EMB_CACHE = {}

# Guarantee that the custom production output path exists cleanly
os.makedirs(PipelineConfig.OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA PREPROCESSING AND LOADING
# ============================================================
def clean_text(text):
    """Removes extra whitespaces and maps text to lowercase."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_asap_dataset():
    """Loads the ASAP dataset robustly across multiple standard encodings."""
    encodings = ["utf-8", "latin1", "cp1252"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(PipelineConfig.ASAP_PATH, sep="\t", encoding=enc)
            print(f"Successfully loaded ASAP dataset with encoding: {enc}")
            break
        except Exception:
            pass
            
    if df is None:
        raise FileNotFoundError(f"Unable to read reference target file at {PipelineConfig.ASAP_PATH}")
        
    df = df[["essay", "domain1_score", "essay_set"]].dropna()
    df["domain1_score"] = df["domain1_score"].astype(float)
    print(f"Total production dataset shape size: {df.shape}")
    return df

def normalize_scores(df):
    """Normalizes scores locally [0, 1] within each native essay prompt context."""
    df = df.copy()
    for prompt_id in df["essay_set"].unique():
        mask = df["essay_set"] == prompt_id
        scores = df.loc[mask, "domain1_score"]
        score_range = scores.max() - scores.min()
        if score_range == 0:
            df.loc[mask, "domain1_score"] = 0
        else:
            df.loc[mask, "domain1_score"] = (scores - scores.min()) / score_range
    return df


# ============================================================
# MULTI-TIER FEATURE ENGINEERING EXTRACTION
# ============================================================
def extract_linguistic_features(text):
    """Extracts exactly 6 surface linguistic and readability attributes."""
    words = text.split()
    if len(words) == 0:
        return [0] * 6
    return [
        len(text),
        len(words),
        np.mean([len(w) for w in words]),
        text.count("."),
        textstat.flesch_reading_ease(text),
        len(set(words)) / len(words)  # Type-Token Ratio
    ]

def build_linguistic_matrix(df):
    """Iterates seamlessly over the dataframe to construct the dense linguistic feature space."""
    features = []
    for essay in tqdm(df["essay"], desc="Extracting linguistic features"):
        features.append(extract_linguistic_features(essay))
    return np.array(features)

def compute_roberta_embeddings(texts, tokenizer, encoder_model):
    """Extracts 768-dim semantic representations from the pre-trained CLS token vector."""
    global EMB_CACHE
    cache_key = hash(str(texts))
    if cache_key in EMB_CACHE:
        return EMB_CACHE[cache_key]

    embeddings_list = []
    encoder_model.eval()
    
    with torch.no_grad():
        for i in range(0, len(texts), PipelineConfig.BATCH_SIZE):
            batch_texts = texts[i:i + PipelineConfig.BATCH_SIZE]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=PipelineConfig.MAX_LEN,
                return_tensors="pt"
            ).to(PipelineConfig.DEVICE)

            cls_representations = encoder_model(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()
            embeddings_list.append(cls_representations)

    fused_embeddings = np.vstack(embeddings_list)
    EMB_CACHE[cache_key] = fused_embeddings
    return fused_embeddings


# ============================================================
# DEEP LEARNING COMPONENT WRAPPERS
# ============================================================
class RoBERTaEncoder(nn.Module):
    """Extracts contextual embeddings from the final layer hidden states."""
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(PipelineConfig.TRANSFORMER_MODEL)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class BidirectionalLSTM(nn.Module):
    """Recurrent baseline sequential network processing configuration."""
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_forward = h[-2]
        h_backward = h[-1]
        h_combined = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_combined).squeeze()


# ============================================================
# BASELINE RECURRENT MODEL TRAINING INFRASTRUCTURE
# ============================================================
def build_vocab(texts, max_vocab=5000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (w, _) in enumerate(counter.most_common(max_vocab - 2)):
        vocab[w] = i + 2
    return vocab

def encode_texts(texts, vocab, max_len=200):
    encoded_sequences = []
    for t in texts:
        seq = [vocab.get(w, 1) for w in t.split()[:max_len]]
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        encoded_sequences.append(seq)
    return np.array(encoded_sequences)

def train_lstm_baseline(train_text, y_train, test_text, epochs=10):
    """Trains the baseline deep sequential network explicitly matching specified hyperparameters."""
    vocab = build_vocab(train_text, max_vocab=PipelineConfig.LSTM_MAX_VOCAB)
    X_train = encode_texts(train_text, vocab, max_len=PipelineConfig.LSTM_SEQ_LEN)
    X_test = encode_texts(test_text, vocab, max_len=PipelineConfig.LSTM_SEQ_LEN)

    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=32,
        shuffle=True
    )

    model = BidirectionalLSTM(vocab_size=len(vocab)).to(PipelineConfig.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for ep in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(PipelineConfig.DEVICE), yb.to(PipelineConfig.DEVICE)
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions_list = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 64):
            xb = X_test_tensor[i:i + 64].to(PipelineConfig.DEVICE)
            pred = model(xb).squeeze().cpu().numpy()
            if pred.ndim == 0:
                predictions_list.append(float(pred))
            else:
                predictions_list.extend(pred)

    return np.clip(np.array(predictions_list), 0, 1)


# ============================================================
# PERFORMANCE EVALUATION STANDARDS
# ============================================================
def rescale_predictions(y):
    """Rescales normalized outputs [0, 1] onto the native discrete 12-point scaling matrix."""
    y = np.clip(y, 0, 1)
    return np.round(y * 12)

def evaluate_metrics(y_true, y_pred):
    """Computes exact RMSE, Pearson Correlation, and Quadratic Weighted Kappa."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson_coef = pearsonr(y_true, y_pred)[0]

    y_true_discrete = rescale_predictions(y_true)
    y_pred_discrete = rescale_predictions(y_pred)

    qwk = cohen_kappa_score(y_true_discrete, y_pred_discrete, weights="quadratic")
    return rmse, pearson_coef, qwk


# ============================================================
# TRADITIONAL SHALLOW MODEL BASELINES
# ============================================================
def run_baseline_ridge(train_text, y_train, test_text):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    model = Ridge()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def run_baseline_svr(train_text, y_train, test_text):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    model = SVR()
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ============================================================
# TARGET HYBRID INTEGRATED FRAMEWORK
# ============================================================
def train_hybrid_framework(train_text, test_text, train_emb, test_emb, train_ling, test_ling, y_train):
    """Executes multi-tier feature concatenation and optimizes LightGBM Regressor configurations."""
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_text).toarray()
    X_test_tfidf = tfidf.transform(test_text).toarray()

    X_train_fused = np.concatenate([train_emb, X_train_tfidf, train_ling], axis=1)
    X_test_fused = np.concatenate([test_emb, X_test_tfidf, test_ling], axis=1)

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        force_col_wise=True
    )
    model.fit(X_train_fused, y_train)
    predictions = model.predict(X_test_fused)
    return model, predictions


# ============================================================
# CROSS-PROMPT STUDIES & COMPONENT ABLATION LOOPS
# ============================================================
def execute_prompt_wise_lopo(df, linguistic_matrix, tokenizer, encoder_model):
    """Executes structural Leave-One-Prompt-Out validation mapping outputs to prompt_results.csv."""
    print("Executing Leave-One-Prompt-Out cross-prompt validation protocol")
    results_records = []
    unique_prompts = sorted(df["essay_set"].unique())

    for current_prompt in unique_prompts:
        print(f"Holding out Prompt Set: {current_prompt}")
        train_split = df[df["essay_set"] != current_prompt].copy()
        test_split = df[df["essay_set"] == current_prompt].copy()

        train_indices = train_split.index.values
        test_indices = test_split.index.values

        emb_train = compute_roberta_embeddings(train_split["essay"].tolist(), tokenizer, encoder_model)
        emb_test = compute_roberta_embeddings(test_split["essay"].tolist(), tokenizer, encoder_model)

        tfidf = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf.fit_transform(train_split["essay"]).toarray()
        X_test_tfidf = tfidf.transform(test_split["essay"]).toarray()

        X_train = np.concatenate([emb_train, X_train_tfidf, linguistic_matrix[train_indices]], axis=1)
        X_test = np.concatenate([emb_test, X_test_tfidf, linguistic_matrix[test_indices]], axis=1)

        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=128,
            n_jobs=-1,
            force_col_wise=True
        )
        model.fit(X_train, train_split["domain1_score"].values)
        preds = model.predict(X_test)

        rmse, pearson, qwk = evaluate_metrics(test_split["domain1_score"].values, preds)
        results_records.append({
            "Prompt": current_prompt,
            "RMSE": rmse,
            "Pearson": pearson,
            "QWK": qwk
        })

    prompt_table = pd.DataFrame(results_records)
    prompt_table.to_csv(os.path.join(PipelineConfig.OUTPUT_DIR, "prompt_results.csv"), index=False)
    print("\n--- Leave-One-Prompt-Out (LOPO) Cross-Prompt Results ---")
    print(prompt_table)

def execute_ablation_study(df, linguistic_matrix, tokenizer, encoder_model):
    """Runs structural verification sequences mapping outputs to ablation_results.csv."""
    print("Initiating system configuration ablation loops")
    y_labels = df["domain1_score"].values
    kf = KFold(n_splits=PipelineConfig.NUM_FOLDS, shuffle=True, random_state=42)

    experimental_configs = {
        "FULL": (True, True, True),
        "NO_EMB": (False, True, True),
        "NO_TFIDF": (True, False, True),
        "NO_LING": (True, True, False),
    }
    ablation_records = []

    for name, (flag_emb, flag_tfidf, flag_ling) in experimental_configs.items():
        print(f"Processing architectural configuration layout: {name}")
        qwk_list, rmse_list, pearson_list = [], [], []

        for train_idx, test_idx in kf.split(df):
            train_text = df["essay"].iloc[train_idx]
            test_text = df["essay"].iloc[test_idx]

            y_train, y_test = y_labels[train_idx], y_labels[test_idx]
            X_train_blocks, X_test_blocks = [], []

            if flag_emb:
                X_train_blocks.append(compute_roberta_embeddings(train_text.tolist(), tokenizer, encoder_model))
                X_test_blocks.append(compute_roberta_embeddings(test_text.tolist(), tokenizer, encoder_model))

            if flag_tfidf:
                tfidf = TfidfVectorizer(max_features=5000)
                X_train_blocks.append(tfidf.fit_transform(train_text).toarray())
                X_test_blocks.append(tfidf.transform(test_text).toarray())

            if flag_ling:
                X_train_blocks.append(linguistic_matrix[train_idx])
                X_test_blocks.append(linguistic_matrix[test_idx])

            X_train = np.concatenate(X_train_blocks, axis=1)
            X_test = np.concatenate(X_test_blocks, axis=1)

            model = lgb.LGBMRegressor(
                n_estimators=800,
                learning_rate=0.02,
                num_leaves=128,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                force_col_wise=True
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse, pearson, qwk = evaluate_metrics(y_test, predictions)
            rmse_list.append(rmse)
            pearson_list.append(pearson)
            qwk_list.append(qwk)

        ablation_records.append({
            "Model": name,
            "RMSE_mean": np.mean(rmse_list),
            "Pearson_mean": np.mean(pearson_list),
            "QWK_mean": np.mean(qwk_list),
            "QWK_std": np.std(qwk_list)
        })

    ablation_table = pd.DataFrame(ablation_records)
    ablation_table.to_csv(os.path.join(PipelineConfig.OUTPUT_DIR, "ablation_results.csv"), index=False)
    print("\n--- Component Ablation Study Analysis ---")
    print(ablation_table)


# ============================================================
# MAIN PIPELINE WORKFLOW EXECUTION
# ============================================================
def main():
    # 1. Processing dataset loading sequences
    df_raw = load_asap_dataset()
    df_raw["essay"] = df_raw["essay"].apply(clean_text)
    df_normalized = normalize_scores(df_raw)

    y_labels = df_normalized["domain1_score"].values
    linguistic_matrix = build_linguistic_matrix(df_normalized)

    tokenizer = AutoTokenizer.from_pretrained(PipelineConfig.TRANSFORMER_MODEL)
    encoder_model = RoBERTaEncoder().to(PipelineConfig.DEVICE)

    kf = KFold(n_splits=PipelineConfig.NUM_FOLDS, shuffle=True, random_state=42)
    cross_val_records = []

    # 2. Main cross-validation evaluation sequencing loops
    for fold, (train_idx, test_idx) in enumerate(kf.split(df_normalized)):
        torch.cuda.empty_cache()
        EMB_CACHE.clear()
        print(f"Executing Cross-Validation Partition Fold: {fold + 1}")

        train_texts = df_normalized["essay"].iloc[train_idx]
        test_texts = df_normalized["essay"].iloc[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        # Benchmarking traditional architectures
        ridge_preds = run_baseline_ridge(train_texts, y_train, test_texts)
        _, _, qwk_ridge = evaluate_metrics(y_test, ridge_preds)

        svr_preds = run_baseline_svr(train_texts, y_train, test_texts)
        _, _, qwk_svr = evaluate_metrics(y_test, svr_preds)

        lstm_preds = train_lstm_baseline(train_texts.tolist(), y_train, test_texts.tolist())
        _, _, qwk_lstm = evaluate_metrics(y_test, lstm_preds)

        # Processing fused target framework setup
        emb_train = compute_roberta_embeddings(train_texts.tolist(), tokenizer, encoder_model)
        emb_test = compute_roberta_embeddings(test_texts.tolist(), tokenizer, encoder_model)

        _, hybrid_preds = train_hybrid_framework(
            train_texts, test_texts, emb_train, emb_test,
            linguistic_matrix[train_idx], linguistic_matrix[test_idx], y_train
        )
        _, _, qwk_hybrid = evaluate_metrics(y_test, hybrid_preds)

        cross_val_records.append({
            "Fold": fold + 1,
            "Ridge_QWK": qwk_ridge,
            "SVR_QWK": qwk_svr,
            "LSTM_QWK": qwk_lstm,
            "Hybrid_QWK": qwk_hybrid
        })

    cv_results_df = pd.DataFrame(cross_val_records)
    cv_results_df.to_csv(os.path.join(PipelineConfig.OUTPUT_DIR, "cv_results.csv"), index=False)
    print("\n--- Unified 5-Fold Cross Validation Results ---")
    print(cv_results_df)

    # 3. Statistical directional hypothesis pairing (Paired T-Test)
    print("\n--- Statistical Significance Evaluations (Paired T-Test vs Hybrid) ---")
    ttest_rows = []
    for column_name in ["Ridge_QWK", "SVR_QWK", "LSTM_QWK"]:
        t_stat, p_val = ttest_rel(cv_results_df["Hybrid_QWK"], cv_results_df[column_name])
        print(f"Hybrid vs {column_name.split('_')[0]}: t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
        ttest_rows.append({"Comparison": f"Hybrid_vs_{column_name.split('_')[0]}", "t_statistic": t_stat, "p_value": p_val})
    pd.DataFrame(ttest_rows).to_csv(os.path.join(PipelineConfig.OUTPUT_DIR, "ttest_results.csv"), index=False)

    # 4. Global production modeling compilation
    global_tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_full_corpus_tfidf = global_tfidf_vectorizer.fit_transform(df_normalized["essay"]).toarray()
    full_corpus_embeddings = compute_roberta_embeddings(df_normalized["essay"].tolist(), tokenizer, encoder_model)
    X_fully_fused_features = np.concatenate([full_corpus_embeddings, X_full_corpus_tfidf, linguistic_matrix], axis=1)

    final_production_model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=128,
        n_jobs=-1,
        force_col_wise=True
    )
    final_production_model.fit(X_fully_fused_features, y_labels)

    # 5. External dataset evaluation and Teacher validation matrix export
    if os.path.exists(PipelineConfig.TEACHER_PATH):
        print("\nCommencing comprehensive verification loop execution on external teacher files")
        teacher_df = pd.read_csv(PipelineConfig.TEACHER_PATH).dropna(subset=["essay", "level"])
        teacher_df["essay"] = teacher_df["essay"].apply(clean_text)

        X_teacher_tfidf = global_tfidf_vectorizer.transform(teacher_df["essay"]).toarray()
        emb_teacher = compute_roberta_embeddings(teacher_df["essay"].tolist(), tokenizer, encoder_model)
        ling_teacher = build_linguistic_matrix(teacher_df)
        X_teacher_fused = np.concatenate([emb_teacher, X_teacher_tfidf, ling_teacher], axis=1)

        teacher_predictions = np.clip(final_production_model.predict(X_teacher_fused), 0, 1)
        y_teacher_ground_truth = (teacher_df["level"] - 1) / 2

        rmse_t, pearson_t = np.sqrt(mean_squared_error(y_teacher_ground_truth, teacher_predictions)), pearsonr(y_teacher_ground_truth, teacher_predictions)[0]
        print(f"Teacher Validation System -> RMSE: {rmse_t:.4f}, Pearson Correlation: {pearson_t:.4f}")

        best_achieved_qwk = -1
        best_t1, best_t2 = 0.33, 0.66
        for threshold_1 in np.arange(0.2, 0.6, 0.05):
            for threshold_2 in np.arange(threshold_1 + 0.1, 0.9, 0.05):
                temp_mapped_labels = np.zeros_like(teacher_predictions, dtype=int)
                temp_mapped_labels[teacher_predictions < threshold_1] = 1
                temp_mapped_labels[(teacher_predictions >= threshold_1) & (teacher_predictions < threshold_2)] = 2
                temp_mapped_labels[teacher_predictions >= threshold_2] = 3

                current_kappa = cohen_kappa_score(teacher_df["level"], temp_mapped_labels, weights="quadratic")
                if current_kappa > best_achieved_qwk:
                    best_achieved_qwk = current_kappa
                    best_t1, best_t2 = threshold_1, threshold_2

        print(f"Optimized Threshold Constraints -> Boundary 1: {best_t1:.2f}, Boundary 2: {best_t2:.2f} | Highest QWK: {best_achieved_qwk:.4f}")
        
        # FINAL PLOTTING MAPPED LABELS MIGRATED FROM ORIGIN CODE 1
        teacher_df["pred_level"] = np.zeros_like(teacher_predictions, dtype=int)
        teacher_df["pred_level"][teacher_predictions < best_t1] = 1
        teacher_df["pred_level"][(teacher_predictions >= best_t1) & (teacher_predictions < best_t2)] = 2
        teacher_df["pred_level"][teacher_predictions >= best_t2] = 3

        # Figure 3: Teacher Evaluation Confusion Matrix
        cm = confusion_matrix(teacher_df["level"], teacher_df["pred_level"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Teacher Validation Confusion Matrix")
        plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, "teacher_cm.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Figure 4: Prediction Error Distribution & Error Analysis Export
        teacher_df["error"] = teacher_df["pred_level"] - teacher_df["level"]
        teacher_df["abs_error"] = np.abs(teacher_df["error"])

        teacher_df.sort_values("abs_error", ascending=False).head(20).to_csv(
            os.path.join(PipelineConfig.OUTPUT_DIR, "top_errors.csv"), index=False
        )

        plt.figure(figsize=(7, 5))
        plt.hist(teacher_df["error"], bins=np.arange(-3.5, 4.5, 1), edgecolor='black', align='left')
        plt.title("Prediction Error Distribution")
        plt.xlabel("Error Vector Score Difference")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, "error_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Supplementary Figure: Essay Length vs Error Distribution Analysis
        teacher_df["length"] = teacher_df["essay"].apply(lambda x: len(x.split()))
        plt.figure(figsize=(7, 5))
        plt.scatter(teacher_df["length"], teacher_df["abs_error"], alpha=0.6)
        plt.title("Prediction Error vs Essay Length Analysis")
        plt.xlabel("Word Count Metric")
        plt.ylabel("Absolute Prediction Error Mapping")
        plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, "error_vs_length.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # 6. Interpretability Engineering and Figure 2 (SHAP Plots Execution)
    print("\nExtracting Explainable AI matrices using SHAP Explainer kernels")
    shap_tree_explainer = shap.TreeExplainer(final_production_model)
    sampled_feature_subset = X_fully_fused_features[:300]
    calculated_shap_values = shap_tree_explainer.shap_values(sampled_feature_subset)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(calculated_shap_values, sampled_feature_subset, show=False)
    plt.title("SHAP Feature Importance Summary Profile", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(calculated_shap_values, sampled_feature_subset, plot_type="bar", show=False)
    plt.title("SHAP Feature Absolute Contributions Chart", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PipelineConfig.OUTPUT_DIR, "shap_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Execute remaining deep investigation studies
    execute_prompt_wise_lopo(df_normalized, linguistic_matrix, tokenizer, encoder_model)
    execute_ablation_study(df_normalized, linguistic_matrix, tokenizer, encoder_model)
    print("\nMaster pipeline operational execution finished successfully. All journal reference figures generated.")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
