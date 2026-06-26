import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def balance_data(X_train: pd.DataFrame, y_train: pd.Series, method: str = 'none', random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies resampling methods to handle class imbalance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        method (str): One of 'none', 'oversample', 'undersample', 'smote', 'smoteenn'.
        random_state (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled features and labels.
    """
    if method == 'none' or method == 'class_weight':
        return X_train, y_train
    elif method == 'oversample':
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_resample(X_train, y_train)
    elif method == 'undersample':
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X_train, y_train)
    elif method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    elif method == 'smoteenn':
        smoteenn = SMOTEENN(random_state=random_state)
        X_res, y_res = smoteenn.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
        
    # Maintain column names
    X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
    y_res_ser = pd.Series(y_res, name=y_train.name)
    return X_res_df, y_res_ser

def get_classifiers(random_state: int = 42, use_class_weight: bool = False, n_jobs: int = 1) -> Dict[str, Any]:
    """
    Returns a dictionary of classifier instances.

    Args:
        random_state (int): Random seed.
        use_class_weight (bool): If True, configures models to use class weights.

    Returns:
        Dict[str, Any]: Model name to estimator mappings.
    """
    cw = 'balanced' if use_class_weight else None
    
    # Classifiers
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, class_weight=cw),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=random_state, class_weight=cw),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight=cw, n_jobs=n_jobs),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight=cw, n_jobs=n_jobs),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, n_jobs=n_jobs,
                                 scale_pos_weight=(67396 / 1258) if use_class_weight else 1.0),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, n_jobs=n_jobs, verbose=-1,
                                  class_weight=cw),
        'CatBoost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_seed=random_state, verbose=0,
                                       auto_class_weights='Balanced' if use_class_weight else None),
        'SGD Classifier': SGDClassifier(loss='log_loss', max_iter=1000, random_state=random_state, class_weight=cw),
        'Linear Support Vector Machine': LinearSVC(C=1.0, dual='auto', random_state=random_state, class_weight=cw)
    }
    
    # Add Voting and Stacking classifiers
    # We select three top models to combine: Logistic Regression, Random Forest, and LightGBM
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=random_state, class_weight=cw)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight=cw, n_jobs=n_jobs)),
        ('lgb', LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, n_jobs=n_jobs, verbose=-1, class_weight=cw))
    ]
    
    models['Voting Classifier'] = VotingClassifier(estimators=estimators, voting='soft')
    
    # Stacking Classifier
    models['Stacking Classifier'] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=random_state, class_weight=cw),
        n_jobs=n_jobs
    )
    
    return models

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a trained model on a test set and returns various metrics.

    Args:
        model: Trained classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        Dict[str, Any]: Metric names mapped to their values.
    """
    # Try predicting probabilities
    has_proba = hasattr(model, "predict_proba")
    
    # LinearSVC doesn't support predict_proba by default. We can use decision_function
    if not has_proba:
        if hasattr(model, "decision_function"):
            dec_func = model.decision_function(X_test)
            # Apply sigmoid to convert to pseudo probabilities
            y_probs = 1 / (1 + np.exp(-dec_func))
            has_proba = True
        else:
            y_probs = None
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
        
    y_pred = model.predict(X_test)
    
    # In case y_pred is numeric class but was float, convert to int
    y_pred = np.round(y_pred).astype(int)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    
    if has_proba and y_probs is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_probs)
        metrics['PR-AUC'] = average_precision_score(y_test, y_probs)
        metrics['Log Loss'] = log_loss(y_test, y_probs)
        metrics['probabilities'] = y_probs
    else:
        metrics['ROC-AUC'] = np.nan
        metrics['PR-AUC'] = np.nan
        metrics['Log Loss'] = np.nan
        metrics['probabilities'] = None
        
    return metrics
