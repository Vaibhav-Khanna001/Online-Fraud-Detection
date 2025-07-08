# Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using a combination of supervised and unsupervised machine learning techniques. The pipeline is designed to address real-world problems like extreme class imbalance and the need for accurate fraud detection with minimal false negatives.

---

## Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Description:** Contains transactions made by European cardholders in September 2013. The dataset includes 284,807 transactions with 492 fraud cases, making it highly imbalanced. All features are anonymized and transformed using PCA except `Time`, `Amount`, and `Class` (target).

---

## Models Used

### Supervised Learning:
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### Unsupervised Learning:
- Isolation Forest  
- One-Class SVM / Local Outlier Factor (LOF) *(Choose based on implementation)*

---

## Workflow

1. Loadeded the dataset into the environment  
2. Checked for missing values and data imbalance  
3. Standardized `Amount` and `Time` features using `StandardScaler`  
4. Handled class imbalance using `RandomUnderSampler`  
5. Trained and evaluated both supervised and unsupervised models  
6. Compared model performance using key metrics  
7. Visualized evaluation metrics for each model

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

- **XGBoost** achieved the best recall and ROC-AUC among the supervised models, making it well-suited for minimizing false negatives in fraud detection.  
- **Isolation Forest** and **One-Class SVM** provided unsupervised alternatives for anomaly detection where labels may be unavailable.  
- Proper feature scaling and undersampling were key to boosting model performance.

---

## Tech Stack

- Python  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`
