# Online Payment Fraud Detection using Machine Learning

This project aims to detect fraudulent online payment transactions using a combination of supervised and unsupervised machine learning models. It addresses real-world challenges such as severe class imbalance and unlabeled anomalies. The goal is to build a robust, interpretable, and scalable fraud detection pipeline.

---

## Dataset

- **Source:** [Kaggle – Online Payments Fraud Detection](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- **File:** `Dataset.csv`
- **Description:** Contains simulated online payment transactions with features like transaction type, amount, origin, destination, and a binary label `isFraud`.

---

## Models Used

### Supervised Learning:
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### Unsupervised Learning:
- Isolation Forest  
- K-Means Clustering  

---

## Workflow

1. Uploaded Kaggle API key to Colab  
2. Downloaded and unzipped the dataset  
3. Preprocessed the data:
   - Dropped ID columns (`nameOrig`, `nameDest`)
   - One-hot encoded the `type` column  
   - Standardized features using `StandardScaler`
4. Addressed class imbalance using `RandomUnderSampler`  
5. Trained and evaluated both supervised and unsupervised models  
6. Compared models using classification metrics  
7. Visualized performance metrics across all models  

---

## Evaluation Metrics

Each model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- PR-AUC  
- Confusion Matrix  

---

## Results Summary

- **XGBoost** achieved the best performance in terms of **Recall** and **ROC-AUC**, making it effective for detecting fraudulent transactions.
- **Isolation Forest** worked well for detecting outliers and anomalies without needing labeled data.
- **K-Means** offered a basic unsupervised approach using cluster-to-label mapping for fraud identification.

---

## Tech Stack

- Python  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`


