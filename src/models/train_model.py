import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')
import yaml

# lr

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Predictions
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# Metrics
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"\nAUC-ROC: {auc_lr:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Bad', 'Good']))

# Confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Bad   Good")
print(f"Actual Bad      [{cm_lr[0,0]:3d}]  [{cm_lr[0,1]:3d}]  → {cm_lr[0,1]} false positives (good predicted as bad)")
print(f"Actual Good     [{cm_lr[1,0]:3d}]  [{cm_lr[1,1]:3d}]  → {cm_lr[1,0]} false negatives (bad predicted as good)")

# lr - class Weights
lr_balanced = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_balanced.fit(X_train, y_train)

y_pred_lr_bal = lr_balanced.predict(X_test)
y_prob_lr_bal = lr_balanced.predict_proba(X_test)[:, 1]

auc_lr_bal = roc_auc_score(y_test, y_prob_lr_bal)
print(f"\nAUC-ROC: {auc_lr_bal:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr_bal, target_names=['Bad', 'Good']))

cm_lr_bal = confusion_matrix(y_test, y_pred_lr_bal)
print(f"Confusion Matrix:")
print(f"                 Bad   Good")
print(f"Actual Bad      [{cm_lr_bal[0,0]:3d}]  [{cm_lr_bal[0,1]:3d}]")
print(f"Actual Good     [{cm_lr_bal[1,0]:3d}]  [{cm_lr_bal[1,1]:3d}]")

# Compare with baseline
print(f"\n--- Comparison with Baseline ---")
print(f"Baseline:     {cm_lr[0,0]} true bad detected, {cm_lr[0,1]} missed")
print(f"Balanced:     {cm_lr_bal[0,0]} true bad detected, {cm_lr_bal[0,1]} missed")
print(f"Improvement:  +{cm_lr_bal[0,0] - cm_lr[0,0]} more bad credits caught")


#random forest

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"\nAUC-ROC: {auc_rf:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Bad', 'Good']))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"Confusion Matrix:")
print(f"                 Bad   Good")
print(f"Actual Bad      [{cm_rf[0,0]:3d}]  [{cm_rf[0,1]:3d}]")
print(f"Actual Good     [{cm_rf[1,0]:3d}]  [{cm_rf[1,1]:3d}]")

#random forest - class Weights