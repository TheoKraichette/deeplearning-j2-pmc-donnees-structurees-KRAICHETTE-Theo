"""
Phase 2 - Baseline PMC régression California Housing.
Sortie Dense(1) SANS activation : un sigmoid bornerait la sortie à [0, 1] alors
que la cible (prix) monte jusqu'à ~5 -> modèle inutilisable.
Loss = mse (guide l'optimiseur), métrique = mae (lisible : 0.5 ≈ 50 000 $).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)


def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


model = build_regression_model(input_dim=8)
model.summary()

history = model.fit(X_train_norm, y_train, epochs=100, batch_size=32,
                    validation_data=(X_val_norm, y_val), verbose=1)

test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")

plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("epoch")
plt.ylabel("MSE (loss)")
plt.legend()
plt.savefig("phase2_loss_curve.png")
