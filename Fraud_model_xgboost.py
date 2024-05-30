import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import json
import joblib
import xgboost as xgb

# Load data
file_path = 'transactions/transactions.txt'
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]
df = pd.DataFrame(data)

print("Data loaded:")
print(df.head())
print(f"Number of samples: {len(df)}")

# Check for empty strings and NaNs
empty_str_columns = df.columns[(df == "").any()].tolist()
nan_columns = df.columns[df.isna().any()].tolist()

print(f"Columns with empty strings: {empty_str_columns}")
print(f"Columns with NaNs: {nan_columns}")

# Replace empty strings with NaNs
df.replace("", float("NaN"), inplace=True)

# Drop columns with a high proportion of NaNs
threshold = 0.5  # Drop columns with more than 50% NaNs
df.dropna(thresh=int(threshold * len(df)), axis=1, inplace=True)

# Drop remaining rows with any NaNs
df.dropna(inplace=True)

print("Data after dropping columns and rows with NaNs:")
print(df.head())
print(f"Number of samples: {len(df)}")

# Convert categorical columns to numeric
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column not in ['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

print("Data after label encoding:")
print(df.head())

# Convert date columns to datetime
df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])
df['accountOpenDate'] = pd.to_datetime(df['accountOpenDate'])
df['dateOfLastAddressChange'] = pd.to_datetime(df['dateOfLastAddressChange'])

print("Data after datetime conversion:")
print(df[['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange']].head())

# Feature engineering: Extracting useful information from dates
df['transactionYear'] = df['transactionDateTime'].dt.year
df['transactionMonth'] = df['transactionDateTime'].dt.month
df['transactionDay'] = df['transactionDateTime'].dt.day
df['accountOpenYear'] = df['accountOpenDate'].dt.year
df['accountOpenMonth'] = df['accountOpenDate'].dt.month
df['accountOpenDay'] = df['accountOpenDate'].dt.day
df['addressChangeYear'] = df['dateOfLastAddressChange'].dt.year
df['addressChangeMonth'] = df['dateOfLastAddressChange'].dt.month
df['addressChangeDay'] = df['dateOfLastAddressChange'].dt.day

# Drop original datetime columns
df.drop(columns=['transactionDateTime', 'accountOpenDate', 'dateOfLastAddressChange'], inplace=True)

print("Data after feature engineering:")
print(df.head())

# Define the features (X) and the target (y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Check the balance of the target variable
print("Target variable distribution:")
print(y.value_counts())

# Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Model evaluation for XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"XGBoost Model score (accuracy): {accuracy_score(y_test, y_pred_xgb)}")
print(f"XGBoost AUC-ROC: {roc_auc_score(y_test, y_pred_proba_xgb)}")
print("XGBoost Classification report:")
print(classification_report(y_test, y_pred_xgb))

print("XGBoost Confusion matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Feature importances for XGBoost
feature_importances_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("XGBoost Feature importances:")
print(feature_importances_xgb.head(10))

# Save the XGBoost model to a file
xgb_model_filename = 'fraud_detection_xgb_model.pkl'
joblib.dump(xgb_model, xgb_model_filename)
print(f"XGBoost model saved to {xgb_model_filename}")

# Optional: Load the saved model
# xgb_model_loaded = joblib.load(xgb_model_filename)
