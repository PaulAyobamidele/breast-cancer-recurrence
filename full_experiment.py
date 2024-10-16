import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from itertools import combinations
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


data = pd.read_csv("BC_dataset(3).csv")

data.info()
data.head()
data.isnull().sum()
type(data)


# plt.figure(figsize=(16, 9))
# sns.pairplot(data=data, hue="Class")
# plt.show()
len(data.columns)


data = data.drop(columns=["Unnamed: 0", "id"])


data["breastquad"] = data["breastquad"].replace("?", np.nan)

feature_names = [
    "Age",
    "Menopause",
    "Tumor_size",
    "No_aux_lymph",
    "Node_caps",
    "Degree_Malignacy",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
    "Target",
]


data.columns = feature_names

len(data.columns)

data.isnull().sum().sort_values(ascending=False)

for col in feature_names:
    print(col, data[col].dtype)


data.head()

data.isnull().sum()


total_entries = data.shape[0]


missing_percentage = (data.isnull().sum() / total_entries) * 100
# Print the missing percentage for each feature
print("Percentage of missing values compared to the total data:")
round(missing_percentage).sort_values(ascending=False)

# Print the visualization of the missing value
missing_data = pd.DataFrame(
    {
        "Feature": missing_percentage.index,
        "Missing Percentage": missing_percentage.values,
    }
).sort_values(by="Missing Percentage", ascending=False)


plt.figure(figsize=(12, 8))
sns.barplot(x="Missing Percentage", y="Feature", data=missing_data, palette="viridis")
plt.title("Percentage of Missing Values by Feature")
plt.xlabel("Percentage of Missing values")
plt.ylabel("Features")
plt.show()

# Find the Categorical Features

categorical_features = [
    "Degree_Malignacy",
    "Age",
    "Menopause",
    "Tumor_size",
    "No_aux_lymph",
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]


for col in feature_names:
    plt.figure(figsize=(10, 5))
    if data[col].dtype == "object" or col in categorical_features:
        sns.countplot(data[col], palette="viridis")
    else:
        sns.histplot(data[col], kde=True, palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.show()


numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = data[numerical_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Data Preprocessing

categorical_columns = [
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]

numerical_features = [
    "Degree_Malignacy",
    "Age",
    "Menopause",
    "Tumor_size",
    "No_aux_lymph",
]


def encode_categorical(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


# Function to train and evaluate XGBoost model
def train_evaluate_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)
    predictions = model.predict(dtest)
    predictions_binary = np.round(predictions)
    accuracy = accuracy_score(y_test, predictions_binary)
    report = classification_report(y_test, predictions_binary)
    return accuracy, report


# Drop rows with missing values in the target column
model_data = data.dropna(subset=["Target"])

# Define categorical features
categorical_features = [
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]

# Identify numerical features
numerical_features = [
    col for col in model_data.columns if col not in categorical_features + ["Target"]
]

### Simple Imputation ###
simple_data = model_data.copy()
mean_imputer = SimpleImputer(strategy="mean")
mode_imputer = SimpleImputer(strategy="most_frequent")

simple_data[categorical_features] = mode_imputer.fit_transform(
    simple_data[categorical_features]
)
simple_data[numerical_features] = mean_imputer.fit_transform(
    simple_data[numerical_features]
)

# Encode categorical variables
simple_data, _ = encode_categorical(simple_data, categorical_features)

# Train and evaluate XGBoost model with simple imputed data
accuracy_simple, report_simple = train_evaluate_xgboost(
    simple_data.drop(columns=["Target"]), simple_data["Target"]
)
print("### Simple Imputation ###")
print(f"Accuracy: {accuracy_simple}")
print("Classification Report:")
print(report_simple)


### KNN Imputation ###
def knn_imputation(model_data):
    knn_data = model_data.copy()
    knn_data, _ = encode_categorical(knn_data, categorical_features)
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_data = knn_imputer.fit_transform(knn_data.drop(columns=["Target"]))
    imputed_df = pd.DataFrame(imputed_data, columns=knn_data.columns[:-1])
    imputed_df["Target"] = knn_data["Target"].values  # Add target back
    return imputed_df


knn_imputed_data = knn_imputation(model_data)

# Train and evaluate XGBoost model with knn imputed data
accuracy_knn, report_knn = train_evaluate_xgboost(
    knn_imputed_data.drop(columns=["Target"]), knn_imputed_data["Target"]
)
print("### KNN Imputation ###")
print(f"Accuracy: {accuracy_knn}")
print("Classification Report:")
print(report_knn)


### Iterative Imputation ###
def iterative_imputation(model_data):
    iterative_data = model_data.copy()
    iterative_data, _ = encode_categorical(iterative_data, categorical_features)
    iterative_imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0
    )
    imputed_data = iterative_imputer.fit_transform(
        iterative_data.drop(columns=["Target"])
    )
    imputed_df = pd.DataFrame(imputed_data, columns=iterative_data.columns[:-1])
    imputed_df["Target"] = iterative_data["Target"].values  # Add target back
    return imputed_df


# Perform iterative imputation
iterative_imputed_data = iterative_imputation(model_data)

# Train and evaluate XGBoost model with iterative imputed data
accuracy_iterative, report_iterative = train_evaluate_xgboost(
    iterative_imputed_data.drop(columns=["Target"]), iterative_imputed_data["Target"]
)
print("### Iterative Imputation ###")
print(f"Accuracy: {accuracy_iterative}")
print("Classification Report:")
print(report_iterative)


### Matrix Factorization (SVD) Imputation ###
def svd_imputation(model_data):
    svd_data = model_data.copy()
    svd_data, _ = encode_categorical(svd_data, categorical_features)
    numerical_imputer = SimpleImputer(strategy="mean")
    svd_data[numerical_features] = numerical_imputer.fit_transform(
        svd_data[numerical_features]
    )
    svd = TruncatedSVD(n_components=5, random_state=42)
    imputed_svd = svd.fit_transform(svd_data[numerical_features])
    for i, col in enumerate(numerical_features):
        svd_data[col] = imputed_svd[:, i]
    return svd_data


svd_imputed_data = svd_imputation(model_data)

accuracy_svd, report_svd = train_evaluate_xgboost(
    svd_imputed_data.drop(columns=["Target"]), svd_imputed_data["Target"]
)
print("### SVD Imputation ###")
print(f"Accuracy: {accuracy_svd}")
print("Classification Report:")
print(report_svd)


# Functions to train and evaluate other models
def train_evaluate_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


def train_evaluate_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


def train_evaluate_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


def train_evaluate_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


def train_evaluate_gradient_boosting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


# Evaluate models on iterative imputed data
X = iterative_imputed_data.drop(columns=["Target"])
y = iterative_imputed_data["Target"]

# Evaluate Logistic Regression
accuracy_lr, report_lr = train_evaluate_logistic_regression(X, y)
print("### Logistic Regression ###")
print(f"Accuracy: {accuracy_lr}")
print("Classification Report:")
print(report_lr)

# Evaluate XGBoost
accuracy_xgb, report_xgb = train_evaluate_xgboost(X, y)
print("### XGBoost ###")
print(f"Accuracy: {accuracy_xgb}")
print("Classification Report:")
print(report_xgb)

# Evaluate Decision Tree
accuracy_dt, report_dt = train_evaluate_decision_tree(X, y)
print("### Decision Tree ###")
print(f"Accuracy: {accuracy_dt}")
print("Classification Report:")
print(report_dt)

# Evaluate Random Forest
accuracy_rf, report_rf = train_evaluate_random_forest(X, y)
print("### Random Forest ###")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:")
print(report_rf)

# Evaluate SVM
accuracy_svm, report_svm = train_evaluate_svm(X, y)
print("### SVM ###")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:")
print(report_svm)

# Evaluate Gradient Boosting
accuracy_gb, report_gb = train_evaluate_gradient_boosting(X, y)
print("### Gradient Boosting ###")
print(f"Accuracy: {accuracy_gb}")
print("Classification Report:")
print(report_gb)


#  Comparing the Models
###################LOGISTIC REGRESSION###########################
def train_evaluate_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


###################DECISION TREE##################################
def train_evaluate_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


#################RANDOM FOREST##################################
def train_evaluate_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


#################SVM##############################################
def train_evaluate_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


################GRADIENT BOOSTING#########################


def train_evaluate_gradient_boosting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


X = iterative_imputed_data.drop(columns=["Target"])
y = iterative_imputed_data["Target"]

# Evaluate Logistic Regression
accuracy_lr, report_lr = train_evaluate_logistic_regression(X, y)
print("### Logistic Regression ###")
print(f"Accuracy: {accuracy_lr}")
print("Classification Report:")
print(report_lr)

# Evaluate XGBoost
accuracy_xgb, report_xgb = train_evaluate_xgboost(X, y)
print("### XGBoost ###")
print(f"Accuracy: {accuracy_xgb}")
print("Classification Report:")
print(report_xgb)

# Evaluate Decision Tree
accuracy_dt, report_dt = train_evaluate_decision_tree(X, y)
print("### Decision Tree ###")
print(f"Accuracy: {accuracy_dt}")
print("Classification Report:")
print(report_dt)

# Evaluate Random Forest
accuracy_rf, report_rf = train_evaluate_random_forest(X, y)
print("### Random Forest ###")
print(f"Accuracy: {accuracy_rf}")
print("Classification Report:")
print(report_rf)

# Evaluate Support Vector Machine
accuracy_svm, report_svm = train_evaluate_svm(X, y)
print("### Support Vector Machine ###")
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:")
print(report_svm)

# Evaluate Gradient Boosting
accuracy_gb, report_gb = train_evaluate_gradient_boosting(X, y)
print("### Gradient Boosting ###")
print(f"Accuracy: {accuracy_gb}")
print("Classification Report:")
print(report_gb)


##############################################
########### VISUALIZATION ####################

models = {
    "Logistic Regression": train_evaluate_logistic_regression,
    "Decision Tree": train_evaluate_decision_tree,
    "Random Forest": train_evaluate_random_forest,
    "Support Vector Machine": train_evaluate_svm,
    "Gradient Boosting": train_evaluate_gradient_boosting,
    "XGBoost": train_evaluate_xgboost,
}

# List of imputation methods
imputation_methods = {
    "Simple Imputation": simple_data,
    "KNN Imputation": knn_imputed_data,
    "Iterative Imputation": iterative_imputed_data,
    "SVD Imputation": svd_imputed_data,
}

# Collect results
results = []

for imp_name, imp_data in imputation_methods.items():
    X = imp_data.drop(columns=["Target"])
    y = imp_data["Target"]
    for model_name, train_evaluate in models.items():
        accuracy, report = train_evaluate(X, y)
        results.append(
            {"Imputation Method": imp_name, "Model": model_name, "Accuracy": accuracy}
        )

# Convert results to a dataframe
results_df = pd.DataFrame(results)


# Bar plot to visualize the average accuracy for each model and imputation method
plt.figure(figsize=(14, 8))
sns.barplot(x="Imputation Method", y="Accuracy", hue="Model", data=results_df, ci="sd")
plt.title("Average Model Accuracy by Imputation Method")
plt.xlabel("Imputation Method")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.show()

heatmap_data = results_df.pivot("Imputation Method", "Model", "Accuracy")

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, fmt=".2f")
plt.title("Heatmap of Model Accuracy by Imputation Method")
plt.xlabel("Model")
plt.ylabel("Imputation Method")
plt.show()


plt.figure(figsize=(14, 8))
sns.lineplot(
    data=results_df, x="Imputation Method", y="Accuracy", hue="Model", marker="o"
)
plt.title("Line Plot of Model Accuracy by Imputation Method")
plt.xlabel("Imputation Method")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.xticks(rotation=45)
plt.show()


sns.pairplot(results_df, hue="Model", height=2.5)
plt.suptitle("Pair Plot of Model Accuracies by Imputation Method", y=1.02)
plt.show()


#######################################Optimizaed Final Model############################################## 

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Load and preprocess data
data = pd.read_csv("BC_dataset(3).csv")
data = data.drop(columns=["Unnamed: 0", "id"])
data["breastquad"] = data["breastquad"].replace("?", np.nan)
feature_names = [
    "Age",
    "Menopause",
    "Tumor_size",
    "No_aux_lymph",
    "Node_caps",
    "Degree_Malignacy",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
    "Target",
]
data.columns = feature_names
categorical_features = [
    "Degree_Malignacy",
    "Age",
    "Menopause",
    "Tumor_size",
    "No_aux_lymph",
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]


# Encode categorical variables
def encode_categorical(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


data, _ = encode_categorical(data, categorical_features)

# Handle missing values
model_data = data.dropna(subset=["Target"])
iterative_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0
)
iterative_imputed_data = iterative_imputer.fit_transform(model_data)
iterative_imputed_df = pd.DataFrame(iterative_imputed_data, columns=model_data.columns)

# Feature engineering
X = iterative_imputed_df.drop(columns=["Target"])
y = iterative_imputed_df["Target"]

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(X_scaled)

# Dimensionality reduction
pca = PCA(n_components=5, random_state=42)
pca_features = pca.fit_transform(poly_features)

# Combine original features with polynomial and PCA features
X_combined = np.hstack((X_scaled, poly_features, pca_features))

# Resampling to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Feature selection using RFE
xgb_estimator = xgb.XGBClassifier(objective="binary:logistic", seed=42)
rfe = RFE(estimator=xgb_estimator, n_features_to_select=20, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Hyperparameter optimization with Grid Search
param_grid = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "scale_pos_weight": [1, 2, 5],
}
grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train_rfe, y_train)

# Best parameters and evaluation
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
best_xgb_model = grid_search.best_estimator_

# Cross-validation for robust performance estimation
cv_scores = cross_val_score(
    best_xgb_model, X_train_rfe, y_train, cv=5, scoring="accuracy"
)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores)}")

y_pred = best_xgb_model.predict(X_test_rfe)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")
print("Classification Report:")
print(report)


# Fitting 5 folds for each of 2304 candidates, totalling 11520 fits
# Best Parameters: {'colsample_bytree': 0.9,
#                   'learning_rate': 0.2,
#                   'max_depth': 4,
#                   'n_estimators': 100,
#                   'scale_pos_weight': 1,
#                   'subsample': 0.8}
# Best Score: 0.7586122448979592
# Cross-Validation Scores: [0.8        0.84       0.82       0.68       0.65306122]
# Mean CV Accuracy: 0.7586122448979592
# Test Accuracy: 0.7936507936507936


# Classification Report:
#               precision    recall  f1-score   support

#          0.0       0.86      0.73      0.79        33
#          1.0       0.74      0.87      0.80        30

#     accuracy                           0.79        63
#    macro avg       0.80      0.80      0.79        63
# weighted avg       0.80      0.79      0.79        63
