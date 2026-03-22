import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/dataset.csv')

print(f"Total data: {len(df)} mahasiswa")
print("\nDistribusi Lulus:")
print(df['lulus_tepat_waktu'].value_counts())

# Prepare features and target
X = df[['ipk', 'absen', 'kegiatan']].copy()
y = df['lulus_tepat_waktu'].copy()

# Encode target variable (Ya=1, Tidak=0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nTarget encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Train Random Forest model
print("\n" + "="*50)
print("Training Random Forest Model...")
print("="*50)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
print("\n" + "="*50)
print("Feature Importance:")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': ['IPK', 'Absen', 'Kegiatan'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Save model and encoder
joblib.dump(model, 'model_random_forest.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\n[OK] Model saved as 'model_random_forest.pkl'")
print("[OK] Label encoder saved as 'label_encoder.pkl'")
