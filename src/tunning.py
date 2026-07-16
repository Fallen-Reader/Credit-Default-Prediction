import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def tune_random_forest(X_train, y_train, cv):
    rf_grid = {
        'n_estimators'     : [100, 200, 300],
        'max_depth'        : [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'max_features'     : ['sqrt', 'log2']
    }
    gs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid,
        n_iter=20,
        cv=cv,
        scoring='roc_auc',
        n_jobs=2,
        random_state=42,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"RF  —> Best AUC : {gs.best_score_:.4f}")
    print(f"RF  —> Best params: {gs.best_params_}")
    return gs.best_estimator_

gs_grid = {
        'n_estimators' : [100, 200, 300],
        'max_depth'    : [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample'    : [0.8, 1.0]
    }
def tune_gradient_boosting(X_train, y_train, cv):
    gs_grid = {
        'n_estimators' : [100, 200, 300],
        'max_depth'    : [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample'    : [0.8, 1.0]
    }
    gs = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gs_grid,
        n_iter=20,
        cv=cv,
        scoring='roc_auc',
        n_jobs=2,
        random_state=42,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"GBT —> Best AUC : {gs.best_score_:.4f}")
    print(f"GBT —> Best params: {gs.best_params_}")
    return gs.best_estimator_


def tune_all(X_train, y_train,config, save_dir="outputs/models/"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    cv_folds = config['tuning']['cv_fold']

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("\nTuning Random Forest...")
    best_rf  = tune_random_forest(X_train, y_train, cv)

    print("\nTuning Gradient Boosting...")
    best_gbt = tune_gradient_boosting(X_train, y_train, cv)

    joblib.dump(best_rf,  f"{save_dir}random_forest_tuned.pkl")
    joblib.dump(best_gbt, f"{save_dir}gradient_boosting_tuned.pkl")
    print(f"\nTuned models saved -> {save_dir}")

    return best_rf, best_gbt

if __name__=="__main__":
        with open("config.yaml") as fs:
            config = yaml.safe_load(fs)
        
        tuned_model_path = config['output']['model_dir']

        df = pd.read_csv(config['data']['train_ds'])
        y_train = df["target"].values.astype(int)
        X_train = df.drop(["target"],axis=1).values.astype(float)

        best_rf,best_gbt = tune_all(X_train, y_train,config=config,save_dir=tuned_model_path)
