import numpy as np
import yaml
import joblib
import pandas as pd
from src.tunning import tune_all
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             recall_score, precision_score, f1_score)

with open("config.yaml") as f:
    config = yaml.safe_load(f)

train_df = pd.read_csv(config['data']['train_ds'])
X_train  = train_df.drop(columns=['target']).values
y_train  = train_df['target'].values

test_df = pd.read_csv(config['data']['test_ds'])
X_test  = test_df.drop(columns=['target']).values
y_test  = test_df['target'].values

best_rf, best_gbt = tune_all(X_train, y_train)


def threshold_analysis(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*65}")
    print(f"  {name} — Threshold Analysis")
    print(f"{'='*65}")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} "
          f"{'F1':>6} {'FN':>5} {'FP':>5}")
    print("-"*55)

    for t in np.arange(0.30, 0.71, 0.05):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        recall    = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1        = f1_score(y_test, y_pred)
        print(f"{t:>10.2f} {recall:>8.3f} {precision:>10.3f} "
              f"{f1:>6.3f} {fn:>5} {fp:>5}")

    print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")


if __name__=='__main__':
    rf = joblib.load(config['tuned_model_path']['Random_forest'])
    gb = joblib.load(config['tuned_model_path']['Gradient_Boosting'])

    threshold_analysis("Random Forest (Tuned)",    rf,  X_test, y_test)
    threshold_analysis("Gradient Boosting (Tuned)", gb, X_test, y_test)