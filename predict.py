
import joblib
import yaml
import argparse

def predict_single(model_name, applicant_data):
   
    model_path = f"outputs/models/{model_name}.pkl"
    model = joblib.load(model_path)

    prob = model.predict_proba(applicant_data)[:, 1][0]
    prediction = model.predict(applicant_data)[0]

    label = "DEFAULT RISK" if prediction == 1 else "CREDITWORTHY"
    print(f"\nModel       : {model_name}")
    print(f"Prediction  : {label}")
    print(f"Probability : {prob:.4f}")
    print(f"Confidence  : {'High' if abs(prob - 0.5) > 0.3 else 'Low'}")

    return prediction, prob

if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    test_df    = pd.read_csv(config['data']['test_ds'])
    applicant  = test_df.drop(columns=['target']).values[[0]]
    actual     = test_df['target'].values[0]
    model_name = ...
    print(f"Actual label: {'Default' if actual == 1 else 'Good'}")

    predict_single("RandomForest_Balanced", applicant)
    predict_single("GradientBoosting", applicant)