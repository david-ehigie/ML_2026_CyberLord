import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Train Random Forest
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 Important Features:\n")
print(feature_importance.head(20))