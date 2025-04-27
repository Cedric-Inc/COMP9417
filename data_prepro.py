import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier



X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").iloc[:, 0]
class_counts = pd.Series(y).value_counts()



missing = X.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    plt.figure(figsize=(10, 4))
    missing.sort_values().plot(kind='bar')
    plt.title("Missing Values per Feature")
    plt.ylabel("Count")
    plt.show()
else:
    print("no miss value")


z = np.abs(zscore(X))
outlier_counts = (z > 3).sum(axis=0)


outliers = pd.Series(outlier_counts, index=X.columns)
outliers = outliers[outliers > 0]


plt.figure(figsize=(10, 4))
outliers.sort_values().tail(20).plot(kind='bar')
plt.title("Outlier Counts per Feature (|Z| > 3)")
plt.xlabel("Feature")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


means = X.mean()
stds = X.std(ddof=0)
upper = means + 3 * stds
lower = means - 3 * stds
X_clip = X.clip(lower=lower, upper=upper, axis=1)
scaler = RobustScaler()
X_clean = pd.DataFrame(scaler.fit_transform(X_clip), columns=X.columns)


f_vals, _ = f_classif(X_clean, y)
mi = mutual_info_classif(X_clean, y, random_state=0)
scores = pd.DataFrame({
    "F_value": f_vals,
    "Mutual_Info": mi
}, index=X.columns)
top_f = scores["F_value"].nlargest(10)
top_mi = scores["Mutual_Info"].nlargest(10)

plt.figure(figsize=(10, 4))
top_f.sort_values().plot(kind='barh')
plt.title("Top 10 Features by ANOVA F-value")
plt.xlabel("F-value")
plt.show()

plt.figure(figsize=(10, 4))
top_mi.sort_values().plot(kind='barh')
plt.title("Top 10 Features by Mutual Information")
plt.xlabel("Mutual Information")
plt.show()


orig_dist = y.value_counts().sort_index()
smote = SMOTE(k_neighbors=3,
              sampling_strategy={c:50 for c, cnt in orig_dist.items() if cnt <= 30},
              random_state=0)
under = RandomUnderSampler(
    sampling_strategy={orig_dist.idxmax(): 2000},
    random_state=0
)
pipe = Pipeline([('smote', smote), ('under', under)])
X_res, y_res = pipe.fit_resample(X_clean, y)
res_dist = y_res.value_counts().sort_index()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
orig_dist.plot(kind='bar')
plt.title("Original Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
res_dist.plot(kind='bar')
plt.title("Resampled Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

class_weights = (1/class_counts).to_dict()
clf = LGBMClassifier(objective='multiclass', num_class=28,
                     class_weight=class_weights,
                     n_estimators=600, learning_rate=0.05,
                     random_state=0)

# Pipeline
pipe = Pipeline([
    ('smote',  smote),
    ('under',  under),
    ('model',  clf)
])






# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
# print("Classification Report:\n")
# print(classification_report(y, y_pred, digits=4))
# print("Macro-F1:", f1_score(y, y_pred, average='macro'))
