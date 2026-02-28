import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("Crop_recommendation.csv")

# STEP 2 — Split features and target
X = data.drop('label', axis=1)
Y = data['label']

# STEP 3 — Encode target labels
le = LabelEncoder()
y = le.fit_transform(Y)

# STEP 4 — Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5 — Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------------------------
# MODELS
# ----------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# To store accuracy for epoch graphs
accuracy_history = {
    "Random Forest": [],
    "Decision Tree": []
}


epochs = 15

for name, model in models.items():
    print(f"\n🔹 Training {name} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        accuracy_history[name].append(acc * 100)
        print(f"Epoch {epoch}: Accuracy = {acc * 100:.2f}%")

    print(f"\n➡ Final Accuracy of {name}: {acc * 100:.2f}%")




best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

joblib.dump(best_model, "crop_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

print("\n✔ Training Completed & Model Saved!")


# SCATTER PLOTS (OBSERVED vs PREDICTED)

dt_model = models["Decision Tree"]
rf_model = models["Random Forest"]

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# GRAPH 1 — Decision Tree Scatter
plt.figure(figsize=(8, 6))
plt.scatter(y_test, dt_pred, color='blue', label="Decision Tree Predictions")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Ideal Fit (y=x)")
plt.xlabel("Observed Crop Labels (Encoded)")
plt.ylabel("Predicted Crop Labels (Encoded)")
plt.title("Observed vs Predicted (Decision Tree)")
plt.legend()
plt.grid(True)
plt.show()

# GRAPH 2 — Random Forest Scatter
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred, color='green', label="Random Forest Predictions")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Ideal Fit (y=x)")
plt.xlabel("Observed Crop Labels (Encoded)")
plt.ylabel("Predicted Crop Labels (Encoded)")
plt.title("Observed vs Predicted (Random Forest)")
plt.legend()
plt.grid(True)
plt.show()

# GRAPH 3 — Comparison Scatter
plt.figure(figsize=(8, 6))
plt.scatter(y_test, dt_pred, color='blue', alpha=0.6, label="Decision Tree")
plt.scatter(y_test, rf_pred, color='green', alpha=0.6, label="Random Forest")
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label="Ideal Fit (y=x)")
plt.xlabel("Observed Crop Labels (Encoded)")
plt.ylabel("Predicted Crop Labels (Encoded)")
plt.title("Observed vs Predicted (DT vs RF)")
plt.legend()
plt.grid(True)
plt.show()

# GRAPH 4 — Random Forest Epoch Accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), accuracy_history["Random Forest"], linewidth=3)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Random Forest Accuracy Across 15 Epochs")
plt.grid(True)
plt.tight_layout()
plt.show()

# GRAPH 5 — Decision Tree Epoch Accuracy
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), accuracy_history["Decision Tree"], linewidth=3)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Decision Tree Accuracy Across 15 Epochs")
plt.grid(True)
plt.tight_layout()
plt.show()

# GRAPH 6 — Comparison Epoch Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), accuracy_history["Random Forest"], label="Random Forest", linewidth=3)
plt.plot(range(1, epochs + 1), accuracy_history["Decision Tree"], label="Decision Tree", linewidth=3)

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison (Random Forest vs Decision Tree)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





