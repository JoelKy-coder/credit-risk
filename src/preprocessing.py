import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    'Total_Amount', 'Total_Amount_to_Repay', 'duration', 
    'Amount_Funded_By_Lender', 'Lender_portion_Funded', 'Lender_portion_to_be_repaid',
    'inflation_rate', 'exchange_rate', 'real_interest_rate', 'average_precipitation', 
    'deposit_interest_rate', 'lending_interest_rate', 'interest_rate_spread', 'unemployment_rate',
    'interest_amount', 'repayment_ratio', 'funding_ratio', 'lender_recovery_ratio',
    'prev_loan_count', 'prev_loan_avg', 'customer_borrowing_freq', 'loan_age',
    'lender_loan_count', 'lender_historical_default_rate'
]

CATEGORICAL_FEATURES = [
    'loan_type', 'New_versus_Repeat', 'lender_id'
]

def get_historical_stats(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes historical stats from the training set to use for test/prediction features.
    This avoids any future data leakage.

    Args:
        train_df (pd.DataFrame): The training set.

    Returns:
        Dict[str, Any]: Lookup tables for customer and lender statistics.
    """
    df = train_df.copy()
    df = df.sort_values('disbursement_date').reset_index(drop=True)
    
    # Compute the cumulative features first to find the final state
    df['prev_loan_count'] = df.groupby('customer_id').cumcount() + 1
    df['prev_loan_sum'] = df.groupby('customer_id')['Total_Amount'].transform(lambda x: x.cumsum())
    df['first_disbursement'] = df.groupby('customer_id')['disbursement_date'].transform('min')
    
    # Customer lookup
    customer_latest = df.sort_values('disbursement_date').groupby('customer_id').last()
    customer_stats = {}
    for cust_id, row in customer_latest.iterrows():
        customer_stats[cust_id] = {
            'prev_loan_count': int(row['prev_loan_count']),
            'prev_loan_sum': float(row['prev_loan_sum']),
            'first_disbursement': row['first_disbursement']
        }
        
    # Lender lookup
    df['lender_loan_count'] = df.groupby('lender_id').cumcount() + 1
    df['lender_default_sum'] = df.groupby('lender_id')['target'].transform(lambda x: x.cumsum())
    lender_latest = df.sort_values('disbursement_date').groupby('lender_id').last()
    
    lender_stats = {}
    for lend_id, row in lender_latest.iterrows():
        lender_stats[lend_id] = {
            'lender_loan_count': int(row['lender_loan_count']),
            'lender_default_sum': float(row['lender_default_sum'])
        }
        
    return {
        'customer_stats': customer_stats,
        'lender_stats': lender_stats
    }

def engineer_features(df: pd.DataFrame, stats: Dict[str, Any] = None, is_train: bool = True) -> pd.DataFrame:
    """
    Engineers date, financial, customer, and aggregated features.

    Args:
        df (pd.DataFrame): Dataset to engineer features on.
        stats (Dict[str, Any]): Historical lookup tables (required if is_train=False).
        is_train (bool): True if training (uses cumulative sums), False if testing/predicting (uses lookup).

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    res = df.copy()
    res['disbursement_date'] = pd.to_datetime(res['disbursement_date'])
    res['due_date'] = pd.to_datetime(res['due_date'])
    
    # 1. Date Features
    res['loan_month'] = res['disbursement_date'].dt.month
    res['loan_year'] = res['disbursement_date'].dt.year
    res['loan_quarter'] = res['disbursement_date'].dt.quarter
    res['loan_weekday'] = res['disbursement_date'].dt.weekday
    
    # 2. Financial Features
    res['interest_amount'] = res['Total_Amount_to_Repay'] - res['Total_Amount']
    res['repayment_ratio'] = res['Total_Amount_to_Repay'] / res['Total_Amount']
    res['funding_ratio'] = res['Amount_Funded_By_Lender'] / res['Total_Amount']
    res['lender_recovery_ratio'] = np.where(
        res['Amount_Funded_By_Lender'] > 0,
        res['Lender_portion_to_be_repaid'] / res['Amount_Funded_By_Lender'],
        0.0
    )
    
    # 3. Customer & Lender Features
    if is_train:
        # Sort chronologically for correct cumulative calculations
        res = res.sort_values('disbursement_date').reset_index(drop=True)
        
        # Customer Features
        res['prev_loan_count'] = res.groupby('customer_id').cumcount()
        res['prev_loan_sum'] = res.groupby('customer_id')['Total_Amount'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        res['prev_loan_avg'] = res['prev_loan_sum'] / res['prev_loan_count'].replace(0, 1)
        res.loc[res['prev_loan_count'] == 0, 'prev_loan_avg'] = 0.0
        
        res['first_disbursement'] = res.groupby('customer_id')['disbursement_date'].transform('min')
        res['loan_age'] = (res['disbursement_date'] - res['first_disbursement']).dt.days
        res['customer_borrowing_freq'] = res['prev_loan_count'] / (res['loan_age'] + 1)
        
        # Lender Features
        res['lender_loan_count'] = res.groupby('lender_id').cumcount()
        res['lender_default_sum'] = res.groupby('lender_id')['target'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        res['lender_historical_default_rate'] = res['lender_default_sum'] / res['lender_loan_count'].replace(0, 1)
        res.loc[res['lender_loan_count'] == 0, 'lender_historical_default_rate'] = 0.0
        
    else:
        # Test or inference mode: look up historical stats
        if stats is None:
            raise ValueError("stats dict must be provided when is_train=False")
            
        cust_stats = stats.get('customer_stats', {})
        lend_stats = stats.get('lender_stats', {})
        
        # Initialize lists to store computed values
        prev_loan_counts = []
        prev_loan_avgs = []
        loan_ages = []
        cust_borrowing_freqs = []
        lender_loan_counts = []
        lender_historical_default_rates = []
        
        for _, row in res.iterrows():
            cust_id = row['customer_id']
            lend_id = row['lender_id']
            disb_date = row['disbursement_date']
            
            # Customer calculations
            if cust_id in cust_stats:
                h = cust_stats[cust_id]
                prev_count = h['prev_loan_count']
                prev_sum = h['prev_loan_sum']
                first_disb = pd.to_datetime(h['first_disbursement'])
                
                # Check if this test loan was disbursed after the training data loans
                # If predicting or test data, we assume these stats are historical
                prev_loan_counts.append(prev_count)
                prev_loan_avgs.append(prev_sum / prev_count if prev_count > 0 else 0.0)
                
                loan_age_days = (disb_date - first_disb).days
                if loan_age_days < 0:
                    loan_age_days = 0
                loan_ages.append(loan_age_days)
                cust_borrowing_freqs.append(prev_count / (loan_age_days + 1))
            else:
                # Brand new customer
                prev_loan_counts.append(0)
                prev_loan_avgs.append(0.0)
                loan_ages.append(0)
                cust_borrowing_freqs.append(0.0)
                
            # Lender calculations
            if lend_id in lend_stats:
                lh = lend_stats[lend_id]
                l_count = lh['lender_loan_count']
                l_def_sum = lh['lender_default_sum']
                
                lender_loan_counts.append(l_count)
                lender_historical_default_rates.append(l_def_sum / l_count if l_count > 0 else 0.0)
            else:
                lender_loan_counts.append(0)
                lender_historical_default_rates.append(0.0)
                
        res['prev_loan_count'] = prev_loan_counts
        res['prev_loan_avg'] = prev_loan_avgs
        res['loan_age'] = loan_ages
        res['customer_borrowing_freq'] = cust_borrowing_freqs
        res['lender_loan_count'] = lender_loan_counts
        res['lender_historical_default_rate'] = lender_historical_default_rates
        
    return res

def build_preprocessing_pipeline() -> Pipeline:
    """
    Creates and returns a Scikit-learn Pipeline with ColumnTransformer for scaling 
    numerical variables and encoding categorical variables.

    Returns:
        Pipeline: Preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    return Pipeline(steps=[('preprocessor', preprocessor)])

def split_and_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline, Dict[str, Any]]:
    """
    Performs train-test split, engineers features on both, and fits the preprocessing pipeline.

    Args:
        df (pd.DataFrame): Cleaned and merged dataset.

    Returns:
        Tuple: X_train_preprocessed, X_test_preprocessed, y_train, y_test, pipeline, training_stats
    """
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Combine training features with target for state extraction
    train_full = X_train.copy()
    train_full['target'] = y_train
    
    # Extract training stats
    stats = get_historical_stats(train_full)
    
    # Feature engineer train and test splits
    X_train_eng = engineer_features(train_full, stats=stats, is_train=True)
    X_test_eng = engineer_features(X_test, stats=stats, is_train=False)
    
    # Fit preprocessing pipeline
    pipeline = build_preprocessing_pipeline()
    X_train_preprocessed = pipeline.fit_transform(X_train_eng[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    X_test_preprocessed = pipeline.transform(X_test_eng[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    
    # Get feature names after one-hot encoding
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + encoded_cat_features
    
    # Convert preprocessed arrays back to DataFrames
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)
    
    return X_train_preprocessed_df, X_test_preprocessed_df, y_train, y_test, pipeline, stats
