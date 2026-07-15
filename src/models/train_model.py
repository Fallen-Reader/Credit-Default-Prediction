import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
import yaml
import importlib
import joblib

with open("config.yaml") as f:
    config = yaml.safe_load(f)


X_train = pd.read_csv(config['data']['train_ds']).drop(columns=['target']).values
y_train = pd.read_csv(config['data']['train_ds'])['target'].values
X_test  = pd.read_csv(config['data']['test_ds']).drop(columns=['target']).values
y_test  = pd.read_csv(config['data']['test_ds'])['target'].values


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm  = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Good (0)', 'Bad (1)'])}")
    print(f"Confusion Matrix:")
    print(f"                     Predicted")
    print(f"                     Good    Bad")
    print(f"Actual Good (0)    [{cm[0,0]:4d}]  [{cm[0,1]:4d}]  ← {cm[0,1]} false alarms")
    print(f"Actual Bad  (1)    [{cm[1,0]:4d}]  [{cm[1,1]:4d}]  ← {cm[1,0]} missed defaults")
    print(f"\nMissed defaults (FN={cm[1,0]}) = loans approved that will default")

    return {
        'model': model,
        'auc':   auc,
        'TP':    cm[1,1], 'TN': cm[0,0],
        'FP':    cm[0,1], 'FN': cm[1,0]
    }

def get_model_class(class_path):
    
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train_all_models(config):
    results = {}
    for model_name, model_config in config['models'].items():
        print(f"\\nTraining {model_name}...")
        
    
        ModelClass = get_model_class(model_config['class'])
        params = model_config['params'].copy()
        
    
        if model_config.get('class_weight'):
            params['class_weight'] = model_config['class_weight']
        
        model = ModelClass(**params)
        

        results[model_name] = evaluate(
            model_name, model,
            X_train, y_train, X_test, y_test
        )

        results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC': [r['auc'] for r in results.values()],
        'TP': [r['TP'] for r in results.values()],
        'TN': [r['TN'] for r in results.values()],
        'FP': [r['FP'] for r in results.values()],
        'FN': [r['FN'] for r in results.values()]
        })

    return results_df

if __name__=='__main__':
    result:dict = train_all_models(config)