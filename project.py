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
data["nodecaps"] = data["nodecaps"].replace("?", np.nan)


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

# Calculate the percentage of missing values for each feature
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


############################################ Model with XGBoost ############################################
############################################ Model with XGBoost ############################################
############################################ Model with XGBoost ############################################

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import TruncatedSVD

xgb_data = data.copy()

# Drop rows where the target is NaN
xgb_data = xgb_data.dropna(subset=["Target"])

print("Missing values before imputation:")
print(xgb_data.isnull().sum())

# Categorical columns to encode
categorical_columns = [
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]

# Label encoding categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    xgb_data[col] = le.fit_transform(
        xgb_data[col].astype(str)
    )  # Convert to string to handle NaNs
    label_encoders[col] = le

# Impute missing values using XGBoost
for col in xgb_data.columns:
    if xgb_data[col].isnull().sum() > 0 and col != "Target":
        print(f"Imputing missing values for: {col}")

        # Create a copy of the data to train the model for the current column
        temp_data = xgb_data.copy()

        # Mask the current column
        temp_data[col] = np.where(temp_data[col].isnull(), np.nan, temp_data[col])

        # Separate data into known and unknown parts
        train_data = temp_data[temp_data[col].notnull()]
        test_data = temp_data[temp_data[col].isnull()]

        # Split into features and target for training
        X_train = train_data.drop(columns=[col, "Target"])
        y_train = train_data[col]

        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)

        # Set parameters for XGBoost to handle missing values
        params = {
            "objective": "reg:squarederror",  # Using regression objective for imputation
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
        }

        # Train a model to impute missing values
        num_rounds = 100
        model = xgb.train(params, dtrain, num_rounds)

        # Predict the missing values
        X_test = test_data.drop(columns=[col, "Target"])
        dtest = xgb.DMatrix(X_test, missing=np.nan)
        imputed_values = model.predict(dtest)

        # Fill the missing values with the predicted values
        xgb_data.loc[xgb_data[col].isnull(), col] = imputed_values

# Now the data is imputed, add the target column back
X = xgb_data.drop("Target", axis=1)
y = xgb_data["Target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost classification
params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Train the model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Make predictions
predictions = model.predict(dtest)
predictions_binary = np.round(predictions)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predictions_binary))


############################################ Model with XGBoost ############################################
############################################ Model with XGBoost ############################################
############################################ Model with XGBoost ############################################


# Define categorical features
categorical_features = [
    "Node_caps",
    "Breast_location",
    "Breast_quad",
    "Radiation_therapy",
]


# Function to encode categorical variables
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

    # Encode categorical variables
    knn_data, _ = encode_categorical(knn_data, categorical_features)

    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_data = knn_imputer.fit_transform(knn_data)
    imputed_df = pd.DataFrame(imputed_data, columns=model_data.columns)
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


### Iterative Imputation ###
def iterative_imputation(model_data):
    iterative_data = model_data.copy()

    iterative_data, _ = encode_categorical(iterative_data, categorical_features)

    iterative_imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0
    )

    # Perform imputation
    imputed_data = iterative_imputer.fit_transform(iterative_data)
    imputed_df = pd.DataFrame(imputed_data, columns=model_data.columns)

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
### SVD Imputation ###
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


# Comparing the Models
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


# Check for Imbalance
target_counts = data["Target"].value_counts()
print("Class distribution:")
print(target_counts)

total_count = len(data)
class_proportions = target_counts / total_count
print("\nClass proportions:")
print(class_proportions)

count = data["Target"].count()
distribution = round(target_counts / count * 100)

# Visualize the distribution
plt.figure(figsize=(8, 6))
bars = target_counts.plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Non-recurrence", "Recurrence"], rotation=0)

# Add legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color="skyblue", ec="k"),
    plt.Rectangle((0, 0), 1, 1, color="salmon", ec="k"),
]
labels = ["Non-recurrence", "Recurrence"]
plt.legend(handles, labels, title="Classes")

plt.show()


##################### MODEL with best imputation, best model and feature extraction and optimization #########################
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


def encode_categorical(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


data, _ = encode_categorical(data, categorical_features)

model_data = data.dropna(subset=["Target"])
iterative_imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), random_state=0
)
iterative_imputed_data = iterative_imputer.fit_transform(model_data)
iterative_imputed_df = pd.DataFrame(iterative_imputed_data, columns=model_data.columns)

X = iterative_imputed_df.drop(columns=["Target"])
y = iterative_imputed_df["Target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(X_scaled)

pca = PCA(n_components=5, random_state=42)
pca_features = pca.fit_transform(poly_features)

X_combined = np.hstack((X_scaled, poly_features, pca_features))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

xgb_estimator = xgb.XGBClassifier(objective="binary:logistic", seed=42)
rfe = RFE(estimator=xgb_estimator, n_features_to_select=20, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

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

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")
best_xgb_model = grid_search.best_estimator_

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


# Null Hypothesis and Alternative Hypothesis


# data_encoded = pd.get_dummies(data, columns=categorical_features)

# target = data["Target"]


# Build a base model without imputation
# Build a model with imputation (mean, median)

# Build an Ensemble model to make comparisons


# Without Handling NANs and empty observations

# Feature Analysis
# X = data.drop(columns="Target")
# y = target


# categorical_features = [
#     "Degree_Malignacy",
#     "Age",
#     "Menopause",
#     "Tumor_size",
#     "No_aux_lymph",
#     "Node_caps",
#     "Breast_location",
#     "Breast_quad",
#     "Radiation_therapy",
# ]

# # Chi square test
# good_features = []

# for feature in feature_names:
#     contingency_table = pd.crosstab(data[feature], data["Target"])
#     chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#     print(f"Chi-square test results for {feature}:")
#     print(f"Chi-square statistic: {chi2}")
#     print(f"P-value: {p_value}")
#     print("")

#     if p_value < 0.05:
#         good_features.append(feature)

# good_features

# chi2_results = {}
# for col in categorical_features:
#     contingency_table = pd.crosstab(data[col], data["Target"])
#     chi2, p, _, _ = chi2_contingency(contingency_table)
#     chi2_results[col] = (chi2, p)
#     print(f"Chi-squared test for {col} vs Target:")
#     print(f"Chi2: {chi2}, p-value: {p}\n")

# # Visualize chi-squared statistics and p-values
# chi2_values = [result[0] for result in chi2_results.values()]
# p_values = [result[1] for result in chi2_results.values()]

# plt.figure(figsize=(12, 6))
# sns.barplot(x=list(chi2_results.keys()), y=chi2_values, palette="viridis")
# plt.title("Chi-squared Statistics for Features vs Target")
# plt.xlabel("Features")
# plt.ylabel("Chi-squared Statistic")
# plt.xticks(rotation=90)
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.barplot(x=list(chi2_results.keys()), y=p_values, palette="viridis")
# plt.axhline(0.05, color="red", linestyle="--")
# plt.title("P-values for Chi-squared Tests of Features vs Target")
# plt.xlabel("Features")
# plt.ylabel("P-value")
# plt.xticks(rotation=90)
# plt.show()
# # ['No_aux_lymph', 'Node_caps', 'Degree_Malignacy', 'Target']


# ############################################ Building Base Model ############################################
# ############################################ Building Base Model ############################################
# ############################################ Building Base Model ############################################

# data.isnull().sum()
# # Drop NAs
# cleaned_data = data_encoded.dropna()

# if cleaned_data.empty:
#     print("DataFrame is empty after dropping missing values.")
# else:
#     X = cleaned_data.drop(columns=["Target"])
#     y = cleaned_data["Target"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     logistic_regression = LogisticRegression(random_state=42)
#     decision_tree = DecisionTreeClassifier(random_state=42)

#     logistic_regression.fit(X_train, y_train)
#     decision_tree.fit(X_train, y_train)

#     y_pred_logistic_regression = logistic_regression.predict(X_test)
#     y_pred_decision_tree = decision_tree.predict(X_test)

#     accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
#     accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

#     print(
#         "Accuracy of Logistic Regression: {:.2f}%".format(
#             accuracy_logistic_regression * 100
#         )
#     )
#     print("Accuracy of Decision Tree: {:.2f}%".format(accuracy_decision_tree * 100))

#     print("\nClassification Report for Logistic Regression:")
#     print(classification_report(y_test, y_pred_logistic_regression))

#     print("\nClassification Report for Decision Tree:")
#     print(classification_report(y_test, y_pred_decision_tree))


# ############################################ Model with Simple Imputation ############################################
# ############################################ Model with Simple Imputation ############################################
# ############################################ Model with Simple Imputation ############################################

# simple_data = data.copy()

# simple_data.isnull().sum()

# simple_data = simple_data.dropna(subset=["Target"])
# simple_data.isnull().sum()

# mean_imputer = SimpleImputer(strategy="mean")
# mode_imputer = SimpleImputer(strategy="most_frequent")
# median_imputer = SimpleImputer(strategy="median")


# simple_data[categorical_features] = mode_imputer.fit_transform(
#     simple_data[categorical_features]
# )

# simple_data.isnull().sum()


# # simple_data[numerical_features] = mean_imputer.fit_transform(
# #     simple_data[numerical_features]
# # )


# simple_data_encoded = pd.get_dummies(simple_data, columns=categorical_features)


# X = simple_data_encoded.drop(columns="Target")
# y = simple_data_encoded["Target"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# logistic_regression = LogisticRegression(random_state=42)
# decision_tree = DecisionTreeClassifier(random_state=42)

# logistic_regression.fit(X_train, y_train)
# decision_tree.fit(X_train, y_train)

# y_pred_logistic_regression = logistic_regression.predict(X_test)
# y_pred_decision_tree = decision_tree.predict(X_test)

# accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
# accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

# print(
#     "Accuracy of Logistic Regression: {:.2f}%".format(
#         accuracy_logistic_regression * 100
#     )
# )
# print("Accuracy of Decision Tree: {:.2f}%".format(accuracy_decision_tree * 100))

# print("\nClassification Report for Logistic Regression:")
# print(classification_report(y_test, y_pred_logistic_regression))

# print("\nClassification Report for Decision Tree:")
# print(classification_report(y_test, y_pred_decision_tree))


# ############################################ Model with KNN Imputation ############################################
# ############################################ Model with KNN Imputation ############################################
# ############################################ Model with KNN Imputation ############################################


# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import LabelEncoder

# knn_data = data.copy()

# knn_data = knn_data.dropna(subset="Target")

# print(knn_data.isnull().sum())


# categorical_columns = [
#     "Node_caps",
#     "Breast_location",
#     "Breast_quad",
#     "Radiation_therapy",
# ]

# label_encoders = {}
# for col in categorical_columns:
#     le = LabelEncoder()
#     knn_data[col] = le.fit_transform(knn_data[col])
#     label_encoders[col] = le


# knn_imputer = KNNImputer(n_neighbors=5)


# imputed_data = knn_imputer.fit_transform(knn_data)

# knn_imputed_df = pd.DataFrame(imputed_data, columns=feature_names)

# knn_imputed_df["Target"].unique()

# for col in feature_names:
#     print(knn_imputed_df[col].dtypes)

# print("Missing values after imputation:\n", knn_imputed_df.isnull().sum())

# print(knn_imputed_df)

# X = knn_imputed_df.drop(columns="Target")
# y = knn_imputed_df["Target"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# logistic_regression = LogisticRegression(random_state=42)
# decision_tree = DecisionTreeClassifier(random_state=42)

# logistic_regression.fit(X_train, y_train)
# decision_tree.fit(X_train, y_train)

# y_pred_logistic_regression = logistic_regression.predict(X_test)
# y_pred_decision_tree = decision_tree.predict(X_test)

# accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
# accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

# print("Logistic Regression Predictions:", y_pred_logistic_regression)
# print("Decision Tree Predictions:", y_pred_decision_tree)

# print(
#     "Accuracy of Logistic Regression: {:.2f}%".format(
#         accuracy_logistic_regression * 100
#     )
# )
# print("Accuracy of Decision Tree: {:.2f}%".format(accuracy_decision_tree * 100))

# print("\nClassification Report for Logistic Regression:")
# print(classification_report(y_test, y_pred_logistic_regression))

# print("\nClassification Report for Decision Tree:")
# print(classification_report(y_test, y_pred_decision_tree))


# ############################################ Model with Iterative Imputation ############################################
# ############################################ Model with Iterative Imputation ############################################
# ############################################ Model with Iterative Imputation ############################################


# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import LabelEncoder

# iterative_data = data.copy()


# iterative_data = iterative_data.dropna(subset=["Target"])

# print(iterative_data.isnull().sum())


# categorical_columns = [
#     "Node_caps",
#     "Breast_location",
#     "Breast_quad",
#     "Radiation_therapy",
# ]

# label_encoders = {}
# for col in categorical_columns:
#     le = LabelEncoder()
#     iterative_data[col] = le.fit_transform(iterative_data[col])
#     label_encoders[col] = le

# iterative_imputer = IterativeImputer(max_iter=10, random_state=0)

# iterative_imputed = iterative_imputer.fit_transform(iterative_data)


# iterative_df = pd.DataFrame(iterative_imputed, columns=feature_names)

# X = iterative_df.drop(columns="Target")
# y = iterative_df["Target"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# logistic_regression = LogisticRegression(random_state=42)
# decision_tree = DecisionTreeClassifier(random_state=42)

# logistic_regression.fit(X_train, y_train)
# decision_tree.fit(X_train, y_train)

# y_pred_logistic_regression = logistic_regression.predict(X_test)
# y_pred_decision_tree = decision_tree.predict(X_test)

# accuracy_logistic_regression = accuracy_score(y_test, y_pred_logistic_regression)
# accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)

# print("Logistic Regression Predictions:", y_pred_logistic_regression)
# print("Decision Tree Predictions:", y_pred_decision_tree)

# print(
#     "Accuracy of Logistic Regression: {:.2f}%".format(
#         accuracy_logistic_regression * 100
#     )
# )
# print("Accuracy of Decision Tree: {:.2f}%".format(accuracy_decision_tree * 100))

# print("\nClassification Report for Logistic Regression:")
# print(classification_report(y_test, y_pred_logistic_regression))

# print("\nClassification Report for Decision Tree:")
# print(classification_report(y_test, y_pred_decision_tree))


# # Voting Classifier


# # logistic_regression_clf = LogisticRegression(random_state=42)
# # decision_tree_clf = DecisionTreeClassifier(random_state=42)
# # random_forest_clf = RandomForestClassifier(random_state=42)
# # voting_classifier = VotingClassifier(
# #     estimators=[
# #         ("lr", logistic_regression_clf),
# #         ("dt", decision_tree_clf),
# #         ("rf", random_forest_clf),
# #     ],
# #     voting="hard",
# # )


# # voting_classifier.fit(X_train_imputed, y_train_imputed)


# # y_pred_voting = voting_classifier.predict(X_test_imputed)

# # accuracy_voting = accuracy_score(y_test_imputed, y_pred_voting)
# # print("Accuracy of Voting Classifier: {:.2f}%".format(accuracy_voting * 100))

# # print("\nClassification Report for Voting Classifier:")
# # print(classification_report(y_test_imputed, y_pred_voting))


# # Other imputation techniques


# # EDA

# for feature in numerical_features:
#     sns.histplot(data[feature], kde=True)
#     plt.title(f"Distribution of {feature}")
#     plt.show()

# for features in categorical_features:
#     sns.countplot(x=data[feature])
#     plt.title(f"Distribution of {feature}")
#     plt.show()


# # for Bivariate

# for feature in categorical_features:
#     sns.boxplot(x=data[feature], hue=data["Target"])
#     plt.title(f"{feature} against {target}")
#     plt.show()


# for feature in numerical_features:
#     sns.boxplot(x=data["Target"], y=data[features])
#     plt.title(f"{feature} versus Target")
#     plt.show()

#     sns.violinplot(x=data["Target"], y=data[feature])
#     plt.title(f"{feature} versus Target")
#     plt.show()


# correlation_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()


# sns.pairplot(data, hue="Target")
# plt.show()


# # BC dataset:
# # “This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia. This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature. (See also lymphography and primary-tumor.)”

# # •	-Class: Target variable indicating whether the event is "no-recurrence" or "recurrence". (Binary)
# # •	-age: Age of the patient. Categories include: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99 years. (Categorical)
# # •	-menopause: Menopausal status of the patient. Categories include: lt40 (less than 40 years), ge40 (greater than or equal to 40 years), premeno (premenopausal). (Categorical)
# # •	-tumor-size: Size of the tumor in millimeters. Categories include: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59. (Categorical)
# # •	-nv-nodes: Number of axillary lymph nodes involved. Categories include ranges like: 0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39. (Categorical)
# # •	-node-caps: Whether there are cancer cells present in the lymph nodes. Categories: yes, no. (Binary)
# # •	-deg-malig: Degree of malignancy of the tumor. Values: 1 (mild), 2 (moderate), 3 (severe). (Integer)
# # •	-breast: Location of the tumor in the breast (left or right). (Binary)
# # •	-breast-quad: Quadrant of the breast where the tumor is located. Categories include: left-up, left-low, right-up, right-low, central. (Categorical)
# # •	-irradiat: Whether the patient received radiation therapy. Categories: yes, no. (Binary)


# # Building Base Models:
# # Start by building baseline models without handling missing values or performing feature selection. This allows you to establish a baseline performance against which you can compare the performance of models with preprocessing and feature selection.
# # Handling Missing Values:
# # Since your dataset contains missing values, you could proceed by handling these missing values using imputation techniques such as mean, median, or mode imputation. After imputation, evaluate the impact on model performance.
# # Building Models with Imputation:
# # Build models using the preprocessed data after handling missing values with imputation. Compare the performance of these models with the baseline models to assess the effectiveness of imputation.
# # Ensemble Modeling:
# # Explore ensemble modeling techniques such as 3 or AdaBoost to create robust models that combine the predictions of multiple base learners. Evaluate the performance of ensemble models and compare them with individual models.
# # Feature Importance Techniques:
# # Utilize feature importance techniques such as decision tree-based algorithms (e.g., Random Forest, Gradient Boosting) to rank features based on their importance in predicting the target variable. Plot feature importance scores to gain insights into the most influential features.
# # Domain Knowledge Analysis:
# # Leverage domain knowledge to further understand which features correlate better with the target variable. Domain-specific insights can guide feature selection and model interpretation.
# # Dimensionality Reduction:
# # Consider dimensionality reduction techniques such as Principal Component Analysis (PCA) or feature selection methods to reduce the number of features while preserving the most relevant information. Evaluate the impact of dimensionality reduction on model performance.


# # Define the problem and objective


# # Research Question/Problem Statement:
# # "Can machine learning models accurately predict the recurrence or non-recurrence of breast cancer based on patient demographic and tumor characteristics such as age, menopausal status, tumor size, number of axillary lymph nodes involved, presence of cancer cells in the lymph nodes, degree of malignancy, tumor location, and whether the patient received radiation therapy?"


# # "How can hyperparameter tuning optimize the performance of machine learning models by selecting and fine-tuning the best features that most significantly predict the recurrence or non-recurrence of breast cancer, based on patient demographic and tumor characteristics?"


# # Exploration / description / diagnosis of data - cleaned data, no outliers, no missing data/values


# # Build your models


# # Hyperparameters


# # Answer your research question


# # Imbalance Ratio for Logistic


# # The results from breast cancer data set depicts that when the number of instances increased from 286 to 699, the percentage of correctly classified instances increased from 69.23% to 96.13% for Random Forest i.e. for dataset with same number of attributes but having more instances, the Random Forest accuracy increased.
