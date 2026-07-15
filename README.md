# Credit Default Prediction with Interpretability

Predict whether a borrower is likely to default on a loan using a classic credit-risk dataset, with a strong focus on model transparency and business-friendly interpretation.

This project uses the **Home Credit Default Risk** dataset from Kaggle, a real-world credit scoring benchmark with multiple tables and borrower-level features. The workflow includes train/test splitting, feature engineering, preprocessing, regularized logistic regression, and interpretability analysis with SHAP and coefficient-based insights.

## Project Overview

Credit default prediction is a high-impact classification problem in lending and risk management.  
Beyond raw prediction performance, this project emphasizes explainability so the results can be translated into actionable business decisions.

### Goals

- Build a reliable credit default classification pipeline.
- Engineer useful borrower and loan features.
- Train baseline and interpretable models.
- Analyze predictions using SHAP values, feature importance, and model coefficients.
- Translate model outputs into business metrics such as risk segments, approval thresholds, and default likelihood.

## Dataset

The project is based on the **Home Credit Default Risk** for now I'm using **German Credit Data** dataset, which includes borrower application data and related credit history tables.


## Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis.
3. Engineer features from application and historical credit data.
4. Build preprocessing pipelines for missing values, encoding, and scaling.
5. Train models such as Logistic Regression with regularization.
6. Compare performance against other baseline classifiers.
7. Explain predictions using SHAP and feature importance.
8. Translate model output into practical lending decisions.

## Models

Planned models for comparison:

- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest

The main focus is on **Logistic Regression** because it offers strong interpretability when combined with regularization and coefficient analysis.

## Interpretability

This project prioritizes explainability at every stage.

Planned interpretability methods:

- SHAP values for local and global explanations.
- Feature importance rankings.
- Logistic regression coefficient analysis.
- Business metric translation for credit decisioning.

Example business questions this project can answer:

- Which features most increase default risk?
- How does a change in income or credit history affect predictions?
- What threshold should be used to flag high-risk borrowers?
- Which borrower segments are safest for approval?

## Repository Structure

```bash

credit-default-prediction/
├── data/
│   ├── raw/                
│   └── processed/          
├── Notebook/
│   └── EDA_01.ipynb         
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models/
│   │   └── train_model.py 
│   └── evaluate.py
├── train.py
├── predict.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Status

🔄 In Progress

- [x] EDA - class balance, distributions, correlation analysis
- [x] Feature engineering
- [x] Preprocessing pipeline
- [x] Model training - Logistic Regression, Naive Bayes, Decision Tree, Random Forest
- [ ] Interpretability — feature importance, coefficient analysis
- [ ] Evaluation — comparison across models

## Metrics

Planned evaluation metrics:

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For imbalanced credit-risk problems, ROC-AUC and recall are often especially useful for assessing default detection quality.

## Setup

> Project in progress - full setup instructions will be added on completion.

**Requirements**

```
pip install -r requirements.txt
```


Or open the notebooks in `notebooks/` to explore the data and model results step by step.

## Dependencies

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap

## Future Improvements

- Hyperparameter tuning.
- Threshold optimization for business use cases.
- Calibration analysis.
- More advanced feature aggregation from auxiliary tables.
- Model comparison with gradient boosting methods.

## License

MIT License

Copyright (c) 2026 Fallen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

