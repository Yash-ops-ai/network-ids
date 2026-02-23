import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("Step 1: Loading dataset...")
files = glob.glob(os.path.join("dataset", "*.csv"))
files = [f for f in files if "plus" not in f]
print(f"Found {len(files)} files: {files}")


dataframes = []
for f in files:
    print(f"Loading {f}...")
    df_temp = pd.read_csv(f)
    dataframes.append(df_temp)


df = pd.concat(dataframes, ignore_index=True)
print(f"Dataset loaded! Total rows: {len(df)}, Total columns: {len(df.columns)}")


print("\nStep 2: Cleaning data...")
df.columns = df.columns.str.strip()
print(f"Checking for null values...")
print(df.isnull().sum().sum())
df.dropna(inplace=True)
print(f"Null values after cleaning: {df.isnull().sum().sum()}")
print(f"Checking for infinite values...")
inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
print(f"Infinite values found: {inf_count}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"Infinite values after cleaning: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")
print(f"Rows remaining after cleaning: {len(df)}")


print("\nStep 3: Encoding labels...")
print(f"Unique labels found: {df['Label'].unique()}")
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"Labels encoded! Mapping:")
for name, number in zip(le.classes_, range(len(le.classes_))):
    print(f"  {number} = {name}")

print("\nStep 4: Separating features and label...")
X = df.drop('Label', axis=1)
y = df['Label']
print(f"Features shape: {X.shape}")
print(f"Label shape: {y.shape}")


print("\nStep 5: Selecting top 20 features...")
from sklearn.ensemble import RandomForestClassifier as RFC
import pandas as pd

selector = RFC(n_estimators=50, random_state=42, n_jobs=-1)
X = X.select_dtypes(include=[np.number])
print(f"Features after removing non-numeric columns: {X.shape[1]}")
selector.fit(X, y)
importances = pd.Series(selector.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index
X = X[top_features]
print(f"Top 20 features selected:")
for i, feature in enumerate(top_features):
    print(f"  {i+1}. {feature}")


print("\nStep 6: Normalizing features...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(f"Normalization complete! All features now between 0 and 1")


print("\nStep 7: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


print("\nStep 8: Training Random Forest model...")
print("This will take several minutes, please wait...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Training complete!")


print("\nStep 9: Evaluating model...")
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


print("\nStep 10: Saving model...")
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("Model saved successfully in models/ folder!")