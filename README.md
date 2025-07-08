Online Payment Fraud Detection using Machine Learning

This project aims to detect fraudulent online payment transactions using a combination of supervised and unsupervised machine learning models. It handles real-world challenges such as severe class imbalance and unlabeled anomalies. The goal is to build a robust and scalable fraud detection pipeline.


Dataset

- Source: (https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- File: `PS_20174392719_1491204439457_log.csv`
- Description: Contains simulated online payment transactions with features like transaction type, amount, origin, destination, and a binary label `isFraud`.


Models Used

Supervised Learning:
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier

Unsupervised Learning:
- Isolation Forest  
- K-Means Clustering


Workflow

1. Uploaded Kaggle API key to Colab
2. Downloaded and unzip dataset
3. Preprocessed data:
   - Dropped IDs (`nameOrig`, `nameDest`)
   - One-hot encode `type` column
   - Standardized features
4. Handled class imbalance using `RandomUnderSampler`
5. Trained & evaluated supervised and unsupervised models
6. Compared models using precision, recall, F1, ROC-AUC, and PR-AUC
7. Made Visualization metrics for all models

---

Evaluation Metrics

Each model is evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix


Results Summary

- XGBoost achieved the best performance in terms of recall and AUC.
- Isolation Forest effectively detected anomalies without label supervision.
- K-Means provided a basic clustering-based fraud separation using post-cluster label mapping.
