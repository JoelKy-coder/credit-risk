import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data(data_dir: str | os.PathLike[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw datasets: train.csv and economic_indicators.csv.

    Args:
        data_dir (str): Directory containing raw files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames of train and economic indicators.
    """
    raw_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data" / "raw"
    train_path = raw_dir / "train.csv"
    econ_path = raw_dir / "economic_indicators.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Raw train data not found at {train_path}")
    if not econ_path.exists():
        raise FileNotFoundError(f"Economic indicators not found at {econ_path}")
        
    train_df = pd.read_csv(train_path)
    econ_df = pd.read_csv(econ_path)
    return train_df, econ_df

def merge_economic_indicators(train_df: pd.DataFrame, econ_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters, pivots, and merges economic indicators for Kenya with the loan training data.

    Args:
        train_df (pd.DataFrame): The loan training dataset.
        econ_df (pd.DataFrame): The economic indicators dataset.

    Returns:
        pd.DataFrame: Merged dataset.
    """
    df = train_df.copy()
    df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
    df['disbursement_year'] = df['disbursement_date'].dt.year

    # Filter for Kenya
    kenya_econ = econ_df[econ_df['Country'] == 'Kenya'].copy()

    indicator_mapping = {
        "Inflation, consumer prices (annual %)": "inflation_rate",
        "Official exchange rate (LCU per US$, period average)": "exchange_rate",
        "Real interest rate (%)": "real_interest_rate",
        "Average precipitation in depth (mm per year)": "average_precipitation",
        "Deposit interest rate (%)": "deposit_interest_rate",
        "Lending interest rate (%)": "lending_interest_rate",
        "Interest rate spread (lending rate minus deposit rate, %)": "interest_rate_spread",
        "Unemployment rate": "unemployment_rate"
    }

    # Map Indicators
    kenya_econ['indicator_name'] = kenya_econ['Indicator'].map(indicator_mapping)
    kenya_econ = kenya_econ.dropna(subset=['indicator_name'])

    # Build year to indicator mapping
    # Years from 2021 to 2024 (2024 forward filled from 2023)
    years = [2021, 2022, 2023, 2024]
    econ_dict = {}
    
    for yr in years:
        col = f"YR{yr}" if yr <= 2023 else "YR2023"
        econ_dict[yr] = {}
        for _, row in kenya_econ.iterrows():
            ind = row['indicator_name']
            val = row[col]
            # Impute average_precipitation for 2022 and 2023 with 2021 value (630.0)
            if ind == "average_precipitation" and pd.isna(val):
                val = 630.0
            econ_dict[yr][ind] = val

    # Add economic columns to df
    for ind in indicator_mapping.values():
        if ind == "fossil_fuel_consumption": # Dropped in mapping
            continue
        df[ind] = df['disbursement_year'].map(lambda y: econ_dict.get(y, econ_dict[2023]).get(ind, np.nan))

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the loan dataset: parses dates, removes duplicates, filters out inconsistencies, 
    and handles missing values.

    Args:
        df (pd.DataFrame): Loan dataset (merged or raw).

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    cleaned_df = df.copy()

    # Convert dates to datetime
    cleaned_df['disbursement_date'] = pd.to_datetime(cleaned_df['disbursement_date'])
    cleaned_df['due_date'] = pd.to_datetime(cleaned_df['due_date'])

    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()

    # Remove critical anomalies:
    # 1. Total_Amount_to_Repay should be >= Total_Amount
    # 2. Amount_Funded_By_Lender should be <= Total_Amount
    anomalous_rows = (
        (cleaned_df['Total_Amount_to_Repay'] < cleaned_df['Total_Amount']) | 
        (cleaned_df['Amount_Funded_By_Lender'] > cleaned_df['Total_Amount'])
    )
    
    cleaned_df = cleaned_df[~anomalous_rows]
    return cleaned_df

def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves a DataFrame to a CSV file in the processed data folder.

    Args:
        df (pd.DataFrame): DataFrame to save.
        filepath (str): Target path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Loads a processed dataset from a CSV file.

    Args:
        filepath (str): Path to CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file not found at {filepath}")
    return pd.read_csv(filepath)
