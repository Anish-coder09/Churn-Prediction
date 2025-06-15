import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load preprocessed data
X = pd.read_csv("preprocessed_features.csv")
y = pd.read_csv("preprocessed_target.csv")['churned']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate models
best_model = None
best_score = 0
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean():.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Track best model
    if cv_scores.mean() > best_score:
        best_score = cv_scores.mean()
        best_model = model

# Save the best model (XGBoost)
with open("best_churn_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nBest Model Saved: XGBoost")
