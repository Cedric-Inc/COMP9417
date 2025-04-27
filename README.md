# COMP9417 Group Project, Team: BETA
## Project Description
This project focuses on developing a machine learning solution to classify customer feedback across 28 different products into corresponding departments.  
The dataset consists of 10,000 training instances and 300 extracted features, primarily utilizing NLP techniques.  
The key objective is to build a robust **multiclass classification** system capable of handling both **class imbalance** and **distribution shifts**.

---

## Project Structure
+ **Data/**: Dataset files (`X_train.csv`, `y_train.csv`)
    - The data is incomplete. For more data, please contact us.
+ **model_lib/**: Model modules
    - `Forest.py`: Random Forest classifier
    - `XGBoostR.py`: XGBoost classifier
    - `Network.py`: Neural Network (MLP) and TabNet models
    - `Tree.py`: Decision Tree baseline model
    - `Ensemble.py`: Voting/Stacking ensemble models
+ **solving_imbalance/**: Sampling methods
    - `Sampling.py`: SMOTE, ADASYN, Tomek Links, Scaling, PCA, etc.
+ **data_prepro.py**: Data preprocessing script
+ **Dis_shifting.py**: Distribution shift simulation (noise injection, masking)
+ **search_para_*.py**:
    - `search_para_Tree.py`, `search_para_RF.py`, `search_para_XGB.py`, `search_para_Neural.py`
    - Hyperparameter tuning scripts for individual models
+ **one_test.py**: Small-scale sample weighting comparison experiment
+ **final_predict.py**: Final ensemble prediction on X_test_1 and X_test_2
+ **README.md**: Project introduction and guide

---

## Functional Modules
+ **Data preprocessing and imbalance handling**
+ **Model training and hyperparameter optimization**
+ **Distribution shift simulation (noise addition, feature masking)**
+ **Model ensembling (Voting and Stacking strategies)**
+ **Comprehensive evaluation across clean and corrupted datasets**

---

## Dependency Environment
+ Python 3.8+
+ scikit-learn
+ xgboost
+ torch
+ pandas
+ numpy
+ imbalanced-learn
+ matplotlib
+ optuna

---

## How to Run
### (1) Preprocessing 
```bash
python data_prepro.py
```

Preprocesses raw input if needed.

### (2) Distribution shift experiments
```bash
python Dis_shifting.py
```

Simulates noisy/masked datasets, trains models, and evaluates robustness.

### (3) Final ensemble prediction
```bash
python final_predict.py
```

Generates final outputs for X_test_1 and X_test_2.

### (4) Model parameter tuning
```bash
python search_para_RF.py
python search_para_XGB.py
python search_para_Neural.py
```

Conducts hyperparameter optimization.

### (5) Sample weighting experiment
```bash
python one_test.py
```

Compares balanced vs effective number reweighting.

---

## Experimental Results
Cross-validation results on the clean training set:

| Indicator | Decision Tree | Random Forest | XGBoost | Neural Net |
| :--- | :--- | :--- | :--- | :--- |
| Balanced Accuracy | 0.21 | 0.45 | 0.46 | 0.52 |
| Recall for Class 5 (major class) | 0.63 | 0.85 | 0.91 | 0.81 |
| Recall for Class 10 (medium class) | 0.36 | 0.45 | 0.82 | 0.73 |
| Recall for Class 17 (minor class) | 0.27 | 0.59 | 0.68 | 0.87 |
| Minimal Class (<10 samples) | All 0 | Part 0 | Still 0 | A few hits (Classes 15,16,18) |
| Entropy Loss | 0.05699 | 0.43275 | 0.00545 | 0.01000 |


---

## Conclusion
+ **Neural Network** achieved the best performance under the original distribution.
+ **Random Forest** demonstrated higher robustness when facing distribution shifts.
+ Among ensembling methods, **VotingClassifier** consistently maintained better stability and lower weighted cross-entropy loss across noisy and masked datasets.
+ Thus, the **VotingClassifier** was selected as the final model for predicting X_test_2.

---

## Author
Ye Zhou  
April 27, 2025

