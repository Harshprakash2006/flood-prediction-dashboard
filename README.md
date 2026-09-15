# Flood Prediction: Beginner-Friendly ML Project

This project predicts `Flood Occurred` (`0` = no flood, `1` = flood) using the actual CSV in `data/`. The original dataset is never changed.

## What it does

- Inspects the source dataset: columns, types, missing values, duplicates, target balance, unique values, and IQR-based unusual-value flags.
- Creates EDA charts for missingness, target balance, correlations, numeric distributions, categorical relationships, and key numeric relationships to the target.
- Trains Logistic Regression, Decision Tree, and Random Forest models.
- Uses a stratified 80/20 train/test split with `random_state=42`.
- Fits imputation, scaling, and one-hot encoding only inside each model pipeline, preventing data leakage.
- Evaluates every model with accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Saves the best model, model-comparison table, detailed quality reports, figures, and Random Forest feature importance.

## Dataset used

The current dataset is `data/flood_risk_dataset_india.csv` (10,000 rows). Its target is `Flood Occurred`. The script automatically selects the single non-empty CSV in `data/`; alternatively, pass an explicit path.

## Setup and run

From the project root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python src/train.py
```

To select a particular CSV:

```powershell
python src/train.py --data data/flood_risk_dataset_india.csv
```

## Interactive user interface

The Streamlit dashboard uses the saved model and includes scenario buttons, a prediction form, input-percentile chart, dataset explorer, model metrics, and feature importance.

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the Local URL printed in the terminal (normally `http://localhost:8501`). Keep the terminal running while using the app; press `Ctrl+C` to stop it.

## Deployment with Streamlit Community Cloud

1. Create a GitHub repository and upload this entire project, including `app.py`, `requirements.txt`, `models/best_flood_model.joblib`, and `data/flood_risk_dataset_india.csv`.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, choose your repository and branch, set the entry point to `app.py`, then select **Deploy**.
4. Streamlit gives you a public `streamlit.app` link to share.

Do not upload private, sensitive, or personally identifiable data to a public repository.

## Results

After a successful run:

- `outputs/data_quality_report.json` — schema and data-quality checks
- `outputs/descriptive_statistics.csv` and `outputs/correlation_matrix.csv` — EDA tables
- `outputs/model_comparison.csv` — metrics for all baselines
- `outputs/confusion_matrix_*.csv` — numeric confusion matrices
- `outputs/random_forest_feature_importance.csv` — all importance values
- `outputs/random_forest_top_10_features.csv` — top ten features
- `outputs/plots/` — EDA, confusion matrix, and feature-importance images
- `models/best_flood_model.joblib` — fitted best-scoring model (selected by ROC-AUC, then F1)

## Notes

The preprocessing pipeline treats every non-numeric feature as categorical and every numeric feature as numeric after inspecting the actual CSV. This avoids relying on hard-coded predictor names. IQR flags are warnings about statistically unusual values, not proof of bad data.
