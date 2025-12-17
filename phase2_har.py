import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, log_loss
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os


def load_signal(file_path):
    """
    Loads a signal file of shape (samples, 128)
    """
    return np.loadtxt(file_path)


np.random.seed(42)
tf.random.set_seed(42)

# Dataset paths
DATASET_DIR = "UCI HAR Dataset"
TRAIN_SIGNALS_DIR = f"{DATASET_DIR}/train/Inertial Signals"
TEST_SIGNALS_DIR = f"{DATASET_DIR}/test/Inertial Signals"

# Train inertial signals
body_acc_x_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_acc_x_train.txt")
body_acc_y_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_acc_y_train.txt")
body_acc_z_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_acc_z_train.txt")

body_gyro_x_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_gyro_x_train.txt")
body_gyro_y_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_gyro_y_train.txt")
body_gyro_z_train = load_signal(f"{TRAIN_SIGNALS_DIR}/body_gyro_z_train.txt")

total_acc_x_train = load_signal(f"{TRAIN_SIGNALS_DIR}/total_acc_x_train.txt")
total_acc_y_train = load_signal(f"{TRAIN_SIGNALS_DIR}/total_acc_y_train.txt")
total_acc_z_train = load_signal(f"{TRAIN_SIGNALS_DIR}/total_acc_z_train.txt")

# Test inertial signals
body_acc_x_test = load_signal(f"{TEST_SIGNALS_DIR}/body_acc_x_test.txt")
body_acc_y_test = load_signal(f"{TEST_SIGNALS_DIR}/body_acc_y_test.txt")
body_acc_z_test = load_signal(f"{TEST_SIGNALS_DIR}/body_acc_z_test.txt")

body_gyro_x_test = load_signal(f"{TEST_SIGNALS_DIR}/body_gyro_x_test.txt")
body_gyro_y_test = load_signal(f"{TEST_SIGNALS_DIR}/body_gyro_y_test.txt")
body_gyro_z_test = load_signal(f"{TEST_SIGNALS_DIR}/body_gyro_z_test.txt")

total_acc_x_test = load_signal(f"{TEST_SIGNALS_DIR}/total_acc_x_test.txt")
total_acc_y_test = load_signal(f"{TEST_SIGNALS_DIR}/total_acc_y_test.txt")
total_acc_z_test = load_signal(f"{TEST_SIGNALS_DIR}/total_acc_z_test.txt")

X_train = np.stack(
    [
        body_acc_x_train,
        body_acc_y_train,
        body_acc_z_train,
        body_gyro_x_train,
        body_gyro_y_train,
        body_gyro_z_train,
        total_acc_x_train,
        total_acc_y_train,
        total_acc_z_train,
    ],
    axis=2,
)

X_test = np.stack(
    [
        body_acc_x_test,
        body_acc_y_test,
        body_acc_z_test,
        body_gyro_x_test,
        body_gyro_y_test,
        body_gyro_z_test,
        total_acc_x_test,
        total_acc_y_test,
        total_acc_z_test,
    ],
    axis=2,
)

print("Train shape:", X_train.shape)  # (samples, 128, 9)
print("Test shape :", X_test.shape)

y_train = np.loadtxt(f"{DATASET_DIR}/train/y_train.txt").astype(int) - 1
y_test = np.loadtxt(f"{DATASET_DIR}/test/y_test.txt").astype(int) - 1

num_classes = len(np.unique(y_train))

scaler = StandardScaler()
# reshape to 2D for scaling
X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

# fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train_2d)
X_test_scaled = scaler.transform(X_test_2d)

# reshape back to 3D
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)


model = Sequential(
    [
        LSTM(100, input_shape=(128, 9), return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(),  # Default learning rate of 0.001
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
)

y_test_proba = model.predict(X_test_scaled)
y_test_pred = np.argmax(y_test_proba, axis=1)

# Save model and scaler
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_lstm.pkl"))
print(f"\nModel saved to {MODELS_DIR}/lstm_model.keras")
print(f"Scaler saved to {MODELS_DIR}/scaler_lstm.pkl")

print("\nLSTM Test Performance")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, average="macro"))
print("Log Loss :", log_loss(y_test, y_test_proba))

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(history.history["loss"], label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("LSTM Training vs Validation Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(history.history["accuracy"], label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("LSTM Training vs Validation Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "lstm_training_curves.png"), dpi=150, bbox_inches="tight"
)
print(f"\nPlot saved to {RESULTS_DIR}/lstm_training_curves.png")
