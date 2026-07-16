import yaml
import pandas as pd
from src.models.train_model import train_all_models

def main():
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_df = pd.read_csv(config['data']['train_ds'])
    test_df  = pd.read_csv(config['data']['test_ds'])

    X_train  = train_df.drop(columns=['target']).values
    y_train  = train_df['target'].values
    X_test   = test_df.drop(columns=['target']).values
    y_test   = test_df['target'].values

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Class balance — Train: {y_train.sum()}/{len(y_train)}")
 
    results_df = train_all_models(config, X_train, y_train, X_test, y_test)

    print("\n" + "="*65)
    print("  FINAL COMPARISON")
    print("="*65)
    print(results_df.to_string(index=False))

    results_df.to_csv(config['output']['result_'], index=False)
    print("\nResults saved -> outputs/results.csv")

if __name__ == '__main__':
    main()