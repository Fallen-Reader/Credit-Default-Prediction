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
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(f"{plots_dir}evaluation.png", dpi=150, bbox_inches='tight')
#plt.show()
plt.close()

print("\nBest Model - Final Report")
print(classification_report(y_test, y_pred,
      target_names=['Good (0)', 'Bad (1)']))

import shap

rf = joblib.load(config['model_path']['random_forest_tuned'])

explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)


# Global bar chart
shap.summary_plot(
    shap_values[:,:,0],
    X_test,
    feature_names=feat_names,
    plot_type="bar",
    show=False          
)
plt.title("SHAP Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig(f"{plots_dir}shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()          
print("Saved -> shap_importance.png")

# Global summary (direction of impact)
shap.summary_plot(
    shap_values[:,:,0],
    X_test,
    feature_names=feat_names,
    show=False
)
plt.title("SHAP Summary — Feature Impact Direction")
plt.tight_layout()
plt.savefig(f"{plots_dir}shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved -> shap_summary.png")

# Local waterfall for a missed default

fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]

if len(fn_indices) > 0:
    fn_idx = fn_indices[0]

    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[:,:,0][fn_idx],
            base_values   = explainer.expected_value[1],
            data          = X_test[fn_idx],
            feature_names = feat_names
        ),
        show=False
    )
    plt.title("Why This Default Was Missed")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}shap_missed_default.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved -> shap_missed_default.png")

# Local waterfall for a correct default catch 
tp_indices = np.where((y_pred == 1) & (y_test == 1))[0]

if len(tp_indices) > 0:
    tp_idx = tp_indices[0]

    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[:,:,0][tp_idx],
            base_values   = explainer.expected_value[1],
            data          = X_test[tp_idx],
            feature_names = feat_names
        ),
        show=False
    )
    plt.title("Why This Default Was Correctly Flagged")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}shap_correct_default.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved -> shap_correct_default.png")

print("\nBusiness Decision Rules (threshold=0.45)")
print("─"*45)
print(f"P(default) < 0.35    →  Auto-approve  ({np.sum(y_prob < 0.35)} applicants)")
print(f"P(default) 0.35 - 0.45 →  Manual review ({np.sum((y_prob >= 0.35) & (y_prob < 0.45))} applicants)")
print(f"P(default) > 0.45    →  Auto-reject   ({np.sum(y_prob >= 0.45)} applicants)")