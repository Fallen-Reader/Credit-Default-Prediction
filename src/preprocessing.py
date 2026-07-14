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
    'sex',
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

    # Nominal - one-hot
    nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=nominal, drop_first=False,dtype=int)

    return df


en_df = encode(df)

numerical_continuous = ['month_duration', 'credit_amount_log', 'loan_burden', 
                        'risk_combo', 'financial_health', 'employment_stability',
                        'loan_to_income_proxy', 'total_credit_exposure', 'age']

def get_continuous_indices(df, config):
    feature_cols = [c for c in df.columns if c != config['data']['target_col']]
    cont_names   = numerical_continuous
    indices      = [feature_cols.index(c) for c in cont_names if c in feature_cols]
    return indices

def normalize(X_train, X_test, continuous_indices):
    mu    = X_train[:, continuous_indices].mean(axis=0)
    sigma = X_train[:, continuous_indices].std(axis=0)
    sigma[sigma == 0] = 1    
    X_train_n = X_train.copy().astype(float)
    X_test_n  = X_test.copy().astype(float)

    X_train_n[:, continuous_indices] = (X_train[:, continuous_indices] - mu) / sigma
    X_test_n[:,  continuous_indices] = (X_test[:,  continuous_indices] - mu) / sigma

    return X_train_n, X_test_n, mu, sigma



def split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n_train = int((1 - test_size) * len(X))
    return (X[idx[:n_train]], X[idx[n_train:]],
            y[idx[:n_train]], y[idx[n_train:]])

def process(config):
    target = config['data']['target_col']
    Y = en_df[target].values.astype(int)
    X = en_df.drop([target],axis=1).values.astype(float)

    testsize,seed = config['split']['test_size'],config['split']['random_seed']

    X_train, X_test, y_train, y_test = split(
        X, Y,
        test_size=testsize,
        seed=seed
    )

    feature_df = df.drop(columns=[target])
    cont_idx   = get_continuous_indices(feature_df, config)

    X_train_n, X_test_n, mu, sigma = normalize(X_train, X_test, cont_idx)

    return X_train_n, X_test_n, y_train, y_test , mu, sigma

if __name__ == "__main__":
    X_train_n, X_test_n, y_train, y_test , mu, sigma=process(config)

    feature_cols = [c for c in en_df.columns if c != 'target']

    X_train_df = pd.DataFrame(X_train_n, columns=feature_cols)
    X_test_df  = pd.DataFrame(X_test_n,  columns=feature_cols)

    X_train_df['target'] = y_train
    X_test_df['target']  = y_test

    X_train_df.to_csv('../Data/processed/train.csv', index=False)
    X_test_df.to_csv('../Data/processed/test.csv',index=False)

    print(f"Train saved: {X_train_df.shape}")
    print(f"Test saved:  {X_test_df.shape}")

