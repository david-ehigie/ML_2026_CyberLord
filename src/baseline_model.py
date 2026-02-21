import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

print("Loading processed data...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Confusion Matrix - Random Forest")
plt.show()