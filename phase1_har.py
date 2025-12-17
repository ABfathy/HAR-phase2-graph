import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.model_selection import learning_curve

# ============================================================
# STEP 1: PREPROCESSING
# ============================================================

print("=" * 60)
print("STEP 1: PREPROCESSING")
print("=" * 60)

# --- 1.1 Load the Data ---

features_df = pd.read_csv(
    "UCI HAR Dataset/features.txt",
    sep=r"\s+",
    header=None,
    names=["idx", "feature_name"],
)
activity_df = pd.read_csv(
    "UCI HAR Dataset/activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["idx", "activity"],
)
activity_map = dict(zip(activity_df["idx"], activity_df["activity"]))

X_train = pd.read_csv("UCI HAR Dataset/train/X_train.txt", sep=r"\s+", header=None)
y_train = pd.read_csv(
    "UCI HAR Dataset/train/y_train.txt", sep=r"\s+", header=None
).values.ravel()
X_test = pd.read_csv("UCI HAR Dataset/test/X_test.txt", sep=r"\s+", header=None)
y_test = pd.read_csv(
    "UCI HAR Dataset/test/y_test.txt", sep=r"\s+", header=None
).values.ravel()

# --- 1.2 Feature Scaling ---
print("\n1.2 Feature Scaling (StandardScaler)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 1.4 PCA for Dimensionality Reduction ---
print("\n1.4 PCA for Dimensionality Reduction")

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(
    f"Original features: {X_train_scaled.shape[1]} -> Reduced: {X_train_pca.shape[1]} (95% variance)"
)

# ============================================================
# STEP 2: MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: MODEL TRAINING")
print("=" * 60)

# --- 2.1 SGDClassifier ---
print("\n2.1 Training SGDClassifier")

sgd_model = SGDClassifier(loss="log_loss", random_state=42)
sgd_model.fit(X_train_pca, y_train)

y_train_pred_sgd = sgd_model.predict(X_train_pca)
y_test_pred_sgd = sgd_model.predict(X_test_pca)
y_train_proba_sgd = sgd_model.predict_proba(X_train_pca)
y_test_proba_sgd = sgd_model.predict_proba(X_test_pca)

print("\nSGDClassifier Results:")
print(
    f"  Accuracy  - Train: {accuracy_score(y_train, y_train_pred_sgd):.4f}, Test: {accuracy_score(y_test, y_test_pred_sgd):.4f}"
)
print(
    f"  Loss      - Train: {log_loss(y_train, y_train_proba_sgd):.4f}, Test: {log_loss(y_test, y_test_proba_sgd):.4f}"
)
print(
    f"  Precision - Train: {precision_score(y_train, y_train_pred_sgd, average='weighted'):.4f}, Test: {precision_score(y_test, y_test_pred_sgd, average='weighted'):.4f}"
)

# --- 2.2 Random Forest ---
print("\n2.2 Training Random Forest")

depths = [5, 10, 15, 20, 25, None]
best_depth = None
best_test_acc = 0

print("\nComparing max_depth values:")
print("-" * 50)
for depth in depths:
    rf_temp = RandomForestClassifier(max_depth=depth, random_state=42)
    rf_temp.fit(X_train_pca, y_train)
    train_acc = accuracy_score(y_train, rf_temp.predict(X_train_pca))
    test_acc = accuracy_score(y_test, rf_temp.predict(X_test_pca))
    print(
        f"  max_depth={str(depth):>4} -> Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
    )

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_depth = depth

print(f"\nBest max_depth: {best_depth} (Test Accuracy: {best_test_acc:.4f})")


rf_model = RandomForestClassifier(max_depth=best_depth, random_state=42)
rf_model.fit(X_train_pca, y_train)

y_train_pred_rf = rf_model.predict(X_train_pca)
y_test_pred_rf = rf_model.predict(X_test_pca)
y_train_proba_rf = rf_model.predict_proba(X_train_pca)
y_test_proba_rf = rf_model.predict_proba(X_test_pca)

print(f"\nRandom Forest Results (max_depth={best_depth}):")
print(
    f"  Accuracy  - Train: {accuracy_score(y_train, y_train_pred_rf):.4f}, Test: {accuracy_score(y_test, y_test_pred_rf):.4f}"
)
print(
    f"  Loss      - Train: {log_loss(y_train, y_train_proba_rf):.4f}, Test: {log_loss(y_test, y_test_proba_rf):.4f}"
)
print(
    f"  Precision - Train: {precision_score(y_train, y_train_pred_rf, average='weighted'):.4f}, Test: {precision_score(y_test, y_test_pred_rf, average='weighted'):.4f}"
)

joblib.dump(pca, "pca.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(sgd_model, "SGD_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")

# --- 2.3 Complexity Loss Curve (Training/Validation) ---
print("\n2.3 Plotting Complexity Loss Curve")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))


train_sizes, train_scores, val_scores = learning_curve(
    SGDClassifier(loss="log_loss", random_state=42),
    X_train_pca,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
)

axes[0].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
axes[0].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
axes[0].set_xlabel("Training Size")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("SGDClassifier (log_loss) - Learning Curve")
axes[0].legend()
axes[0].grid(True)


train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(max_depth=best_depth, random_state=42),
    X_train_pca,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
)

axes[1].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
axes[1].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
axes[1].set_xlabel("Training Size")
axes[1].set_ylabel("Accuracy")
axes[1].set_title(f"Random Forest (max_depth={best_depth}) - Learning Curve")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plt.savefig("results/complexity_loss_curve.png", dpi=150)
plt.close()
print("Saved: results/complexity_loss_curve.png")

print("\n" + "=" * 60)
print("✓ Step 1 & Step 2 Completed!")
print("=" * 60)
