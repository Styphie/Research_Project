# Drug-Target Interaction Prediction

This project implements machine learning models to predict drug–target interactions (DTI) using molecular fingerprints (ECFP4) and protein sequence features (AAC). It is inspired by the paper: _Subpocket-informed transformer for drug–target interaction prediction_.

## Project Structure

- `data_processing.py`: Script to process raw drug–target data into feature matrices.
- `train_evaluate_models.py`: Script to train Logistic Regression, Random Forest, and SVM models.
- `models/`: Folder containing trained models saved as `.joblib` files (tracked with Git LFS).
- `results/`: Folder with evaluation metrics.
- `Research Project/`: Processed datasets (X_train, X_val, X_test, etc.).

## How to Run

1. Clone the repository:

```bash
git clone git@github.com:Styphie/Research_Project.git
cd Research_Project
```
2. Install dependencies (using conda):

```
conda activate dti_project
pip install -r requirements.txt
```
3. Process data:

```
python data_processing.py
```

4. Train models:
```
python train_evaluate_models.py
```

## Models
We implemented and evaluated:

* Logistic Regression

* Random Forest

* Support Vector Machine (SVM)

Performance metrics are saved in `results/evaluation_results.csv.`

## Git LFS
This repo uses Git Large File Storage to track `.joblib` model files.

Please install Git LFS before cloning:

```
git lfs install
git lfs pull
```
## Reference

* Wang et al., *Subpocket-informed transformer for drug–target interaction prediction*, 2023.