import pandas as pd
import numpy as np 
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


df  = pd.read_csv(config['data']['processed_path'])

for col in df.columns :
    print(f"--{col}--")
    print(f"{df[col].value_counts()}")
    print("--"*40)

categorical_nominal = ['status_account', 'credit_history', 'sex', 'marital_status', 
                        'housing', 'job', 'telephone', 'is_foreign_worker', 
                        'other_installment_plans', 'purpose_risk', 'age_group']

categorical_ordinal = ['status_savings', 'years_employment', 'payment_to_income_ratio', 
                        'residence_since']

# Numerical columns 
numerical_continuous = ['month_duration', 'credit_amount_log', 'loan_burden', 
                        'risk_combo', 'financial_health', 'employment_stability',
                        'loan_to_income_proxy', 'total_credit_exposure', 'age']

numerical_binary = ['has_collateral', 'has_guarantor', 'n_credits', 'n_guarantors']

# redundant 
columns_to_drop = ['credit_amount', 'purpose', 'collateral', 'secondary_obligor', 
                   'has_guarantor']  

print(f"\nNominal categorical ({len(categorical_nominal)}): {categorical_nominal}")
print(f"\nOrdinal categorical ({len(categorical_ordinal)}): {categorical_ordinal}")
print(f"\nContinuous numerical ({len(numerical_continuous)}): {numerical_continuous}")
print(f"\nBinary numerical ({len(numerical_binary)}): {numerical_binary}")
print(f"\nColumns to DROP ({len(columns_to_drop)}): {columns_to_drop}")