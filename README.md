![image] (https://www.kaggle.com/datasets/himelsarder/loan-default-risk-prediction-dataset)
# Credit Risk Prediction

End-to-end machine learning project for predicting loan default risk from customer, loan, lender, and macroeconomic indicators. The repository includes data loading, cleaning, exploratory analysis, statistical testing, feature engineering, imbalance handling, model training, model interpretation, saved artifacts, and a Flask prediction app.

## Business Problem

Loan defaults create direct losses for lenders and reduce portfolio stability. This project builds a binary classifier that estimates whether a loan is likely to default so financial teams can prioritize review, pricing, and risk controls.

## Objectives

- Load and validate raw loan and economic indicator data.
- Clean dates, duplicate rows, invalid financial records, missing values, and outliers.
- Explore default behavior across loan, customer, lender, and macroeconomic features.
- Engineer leakage-aware historical customer and lender features.
- Compare imbalance strategies and classification models.
- Tune and save a deployable model and preprocessing pipeline.
- Serve predictions through a Flask API and HTML interface.

## Dataset

The main dataset is `data/raw/train.csv`; macroeconomic indicators are in `data/raw/economic_indicators.csv`.

| Field | Meaning |
| --- | --- |
| `customer_id` | Borrower identifier |
| `tbl_loan_id` | Loan identifier |
| `lender_id` | Lender identifier |
| `Total_Amount` | Amount disbursed |
| `Total_Amount_to_Repay` | Expected repayment amount |
| `loan_type` | Loan product category |
| `disbursement_date`, `due_date` | Loan issue and due dates |
| `duration` | Loan term in days |
| `New_versus_Repeat` | New or repeat borrower flag |
| `target` | `1` for default, `0` for repaid |

The target is highly imbalanced, so accuracy is reported but not treated as the main success metric. Recall, precision, F1, ROC-AUC, and PR-AUC are more relevant for default detection.

## Project Structure

```text
.
|-- app/
|   |-- main.py
|   `-- templates/index.html
|-- data/
|   |-- raw/
|   `-- processed/cleaned_loans.csv
|-- models/
|   |-- best_model.pkl
|   |-- pipeline.pkl
|   |-- training_stats.pkl
|   `-- model_metrics.csv
|-- notebooks/
|   |-- 01_data_loading.ipynb
|   |-- 02_data_cleaning.ipynb
|   |-- 03_eda.ipynb
|   |-- 04_feature_engineering.ipynb
|   |-- 05_model_training.ipynb
|   `-- 06_model_evaluation.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- prediction.py
|   |-- train_pipeline.py
|   |-- training.py
|   `-- utils.py
|-- requirements.txt
`-- prompt.md
```

## Workflow

1. Data loading and validation.
2. Cleaning and economic indicator merge.
3. EDA with univariate, bivariate, and multivariate analysis.
4. Correlation, chi-square, Cramers V, point-biserial correlation, and VIF checks.
5. Date, financial, customer, lender, and aggregated feature engineering.
6. Leakage-aware train-test split and Scikit-learn preprocessing pipeline.
7. Imbalance comparison with class weights, oversampling, undersampling, SMOTE, and SMOTEENN.
8. Model comparison across linear, tree, boosting, voting, and stacking classifiers.
9. LightGBM hyperparameter tuning, evaluation, interpretation, and artifact saving.
10. Flask deployment for JSON and browser-based prediction.

## Model Results

The persisted model is a tuned class-weighted LightGBM classifier.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8757 |
| Precision | 0.0346 |
| Recall | 0.2143 |
| F1-score | 0.0596 |
| ROC-AUC | 0.5439 |
| PR-AUC | 0.0406 |
| Log Loss | 0.5553 |

Best parameters: `learning_rate=0.05`, `max_depth=5`, `num_leaves=31`.

The low precision and PR-AUC show that default prediction remains difficult on this highly imbalanced dataset. The model is suitable as a portfolio triage baseline, not as an automated credit approval system without further validation, threshold tuning, and business-cost analysis.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Training

```bash
python src/train_pipeline.py
```

This command regenerates:

- `data/processed/cleaned_loans.csv`
- `models/best_model.pkl`
- `models/pipeline.pkl`
- `models/training_stats.pkl`
- `models/model_metrics.csv`

## Run the Flask App

```bash
python app/main.py
```

Open `http://127.0.0.1:5000` in a browser.

### JSON Prediction Example

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_id\":\"257244\",\"lender_id\":\"267278\",\"loan_type\":\"Type_1\",\"Total_Amount\":10000,\"Total_Amount_to_Repay\":11000,\"disbursement_date\":\"2024-06-01\",\"due_date\":\"2024-06-08\",\"duration\":7,\"New_versus_Repeat\":\"Repeat Loan\",\"Amount_Funded_By_Lender\":5000,\"Lender_portion_Funded\":0.5,\"Lender_portion_to_be_repaid\":5500}"
```

## Technologies

Python, pandas, NumPy, Matplotlib, Seaborn, SciPy, Scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, SHAP, Flask, Joblib, and Jupyter.

## Deployment Notes

- The app validates required fields and basic loan amount logic before inference.
- The prediction module loads artifacts lazily and applies the same feature engineering and preprocessing used during training.
- Historical customer and lender lookup statistics are saved separately to avoid leakage during inference.
- For production, add authentication, request logging, drift monitoring, calibrated thresholds, and model governance review.

## Future Improvements

- Tune decision thresholds against financial cost of false positives and false negatives.
- Add time-based validation to better simulate future production scoring.
- Expand macroeconomic indicators and customer repayment history.
- Calibrate predicted probabilities before operational use.
- Add automated tests and CI for training, prediction, and Flask routes.

## License

MIT License.

## Author

Joel Kioko
