import pandas as pd
import numpy as np 
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


df  = pd.read_csv(config['data']['processed_path'])


ORDINAL_MAPS = {
    'status_account': {
        'negative': 0,
        'low_balance': 1,
        'no_checking': 2,
        'high_balance': 3
    },
    'status_savings': {
        'unknown': 0,
        'very_low': 1,
        'low':2,
        'medium': 3,
        'high': 4,
    },
    'years_employment': {
        'unemployed': 0,
        'Junior': 1,
        'Mid-Level': 2,
        'Senior': 3,
        'Expert': 4
    },
    'age_group': {
        'young': 0,
        'mid_young': 1,
        'adults': 2
    }
}

BINARY_MAPS = {
    'telephone':        {'none': 0, 'yes': 1},
    'is_foreign_worker': {'yes': 1, 'no': 0},
}

NOMINAL_COLS = [
    'sex'
    'credit_history',
    'other_installment_plans',
    'housing',
    'job',
    'marital_status',
    'purpose_risk'
]


def encode(df):
    df = df.copy()

    # Ordinal
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
            if df[col].isnull().any():
                bad = df[df[col].isnull()][col]
                raise ValueError(f"{col} has unmapped values: {bad.unique()}")

    # Binary
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Nominal — one-hot
    nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=nominal, drop_first=True,dtype=int)

    return df


en_df = encode(df)

print(en_df['credit_history'])