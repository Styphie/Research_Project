# train_evaluate_models.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib
import os

def calculate_metrics(y_true, y_pred_proba, y_pred_binary):
    """Calculates ROC-AUC, PR-AUC, Sensitivity, and Specificity."""
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }

def find_best_threshold(y_true, y_pred_proba):
    """Finds the threshold that maximizes F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # Exclude the last threshold which corresponds to recall=0
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9) # Add epsilon for stability
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

if __name__ == "__main__":
    processed_data_dir = "Research Project"
    results_dir = "results"
    models_dir = "models"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv(f"{processed_data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{processed_data_dir}/y_train.csv").iloc[:, 0] # Get Series
    X_val = pd.read_csv(f"{processed_data_dir}/X_val.csv")
    y_val = pd.read_csv(f"{processed_data_dir}/y_val.csv").iloc[:, 0]
    X_test = pd.read_csv(f"{processed_data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{processed_data_dir}/y_test.csv").iloc[:, 0]

    print("Data loaded.")
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # Scale features (important for LR and SVM)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    # Save the scaler
    joblib.dump(scaler, f"{models_dir}/scaler.joblib")
    print("Scaler saved.")

    results = {}

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0) # Default C, increased max_iter
    lr_model.fit(X_train_scaled, y_train)
    joblib.dump(lr_model, f"{models_dir}/logistic_regression.joblib")
    print("Logistic Regression trained and saved.")

    print("Evaluating Logistic Regression...")
    y_val_pred_proba_lr = lr_model.predict_proba(X_val_scaled)[:, 1]
    best_threshold_lr = find_best_threshold(y_val, y_val_pred_proba_lr)
    print(f"Best LR threshold (Validation F1): {best_threshold_lr}")

    y_test_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_binary_lr = (y_test_pred_proba_lr >= best_threshold_lr).astype(int)
    results["Logistic Regression"] = calculate_metrics(y_test, y_test_pred_proba_lr, y_test_pred_binary_lr)
    print("Logistic Regression evaluated.")

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    # Using scaled data for consistency, though RF is less sensitive to scaling
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Default n_estimators
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, f"{models_dir}/random_forest.joblib")
    print("Random Forest trained and saved.")

    print("Evaluating Random Forest...")
    y_val_pred_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
    best_threshold_rf = find_best_threshold(y_val, y_val_pred_proba_rf)
    print(f"Best RF threshold (Validation F1): {best_threshold_rf}")

    y_test_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_binary_rf = (y_test_pred_proba_rf >= best_threshold_rf).astype(int)
    results["Random Forest"] = calculate_metrics(y_test, y_test_pred_proba_rf, y_test_pred_binary_rf)
    print("Random Forest evaluated.")

    # --- Support Vector Machine (SVM) ---
    print("\nTraining SVM...")
    # Using a smaller subset for SVM training if needed due to computational cost, but try full first.
    # Using probability=True is computationally expensive, needed for ROC/PR AUC and thresholding.
    svm_model = SVC(probability=True, random_state=42, C=1.0, kernel='rbf') # Default C and kernel
    svm_model.fit(X_train_scaled, y_train)
    joblib.dump(svm_model, f"{models_dir}/svm.joblib")
    print("SVM trained and saved.")

    print("Evaluating SVM...")
    y_val_pred_proba_svm = svm_model.predict_proba(X_val_scaled)[:, 1]
    best_threshold_svm = find_best_threshold(y_val, y_val_pred_proba_svm)
    print(f"Best SVM threshold (Validation F1): {best_threshold_svm}")

    y_test_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred_binary_svm = (y_test_pred_proba_svm >= best_threshold_svm).astype(int)
    results["SVM"] = calculate_metrics(y_test, y_test_pred_proba_svm, y_test_pred_binary_svm)
    print("SVM evaluated.")

    # Save results
    results_df = pd.DataFrame(results).T # Transpose to have models as rows
    results_df.index.name = "Model"
    results_file = f"{results_dir}/evaluation_results.csv"
    results_df.to_csv(results_file)
    print(f"\nEvaluation results saved to {results_file}")
    print(results_df)

    print("\nModel training and evaluation complete.")

