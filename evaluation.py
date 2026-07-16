import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             RocCurveDisplay, classification_report)

with open("config.yaml") as f:
    config = yaml.safe_load(f)


test_df = pd.read_csv(config['data']['test_ds'])
X_test  = test_df.drop(columns=['target']).values
y_test  = test_df['target'].values


import os
model_dir = f"{config['output']['model_dir']}"
models    = {}
for fname in os.listdir(model_dir):
    if fname.endswith('.pkl'):
        name = fname.replace('.pkl', '')
        models[name] = joblib.load(f"{model_dir}{fname}")

#ROC Curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for name, model in models.items():
    if 'Balanced' not in name:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        RocCurveDisplay.from_predictions(
            y_test, y_prob,
            name=f"{name} ({auc:.3f})",
            ax=axes[0]
        )
axes[0].plot([0,1], [0,1], 'k--', label='Random')
axes[0].set_title("ROC Curves - All Models")
axes[0].legend(fontsize=7)


# Confusion matrix for best model

best_model = models.get('random_forest_tuned',
                         models.get('random_forest'))
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Good', 'Bad'],
            yticklabels=['Good', 'Bad'],
            ax=axes[1])
axes[1].set_title("Best Model - Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# Feature importance
importances = best_model.feature_importances_
feat_names  = test_df.drop(columns=['target']).columns
indices     = np.argsort(importances)[::-1][:15]

axes[2].barh(range(15), importances[indices][::-1])
axes[2].set_yticks(range(15))
axes[2].set_yticklabels([feat_names[i] for i in indices[::-1]])
axes[2].set_title("Top 15 Feature Importances")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plots_dir = config['output']['plots_dir']
#os.makedirs("outputs/plots", exist_ok=True)
plt.savefig(f"{plots_dir}evaluation.png", dpi=150, bbox_inches='tight')
#plt.show()


print("\nBest Model - Final Report")
print(classification_report(y_test, y_pred,
      target_names=['Good (0)', 'Bad (1)']))