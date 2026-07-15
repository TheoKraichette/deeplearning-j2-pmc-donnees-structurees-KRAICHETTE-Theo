"""
Phase 3 - Diagnostic TensorBoard California Housing.
Deux runs (normalisé vs brut) horodatés : sans horodatage, deux runs écriraient
dans le même dossier et mélangeraient les courbes.
Interprétation des zones en bas de fichier.
"""

import datetime
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


def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_with_tensorboard(X_tr, y_tr, X_va, y_va, run_name, epochs=100):
    log_dir = f"logs/fit/{run_name}_" + datetime.datetime.now().strftime("%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = build_regression_model(input_dim=8)
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                        epochs=epochs, batch_size=32, verbose=0, callbacks=[tb_callback])
    print(f"Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history


model_norm, history_norm = train_with_tensorboard(X_train_norm, y_train, X_val_norm, y_val, "california_norm")
model_raw, history_raw = train_with_tensorboard(X_train, y_train, X_val, y_val, "california_raw")

print(f"norm : val_loss finale = {history_norm.history['val_loss'][-1]:.3f}")
print(f"raw  : val_loss finale = {history_raw.history['val_loss'][-1]:.3f}")

# Visualiser :
#   docker compose exec -d dl tensorboard --logdir=logs/fit --host 0.0.0.0
#   http://localhost:6006

# Interprétation :
# - california_norm : zone (a) sain. train et val descendent ensemble, val (~0.28)
#   juste au-dessus du train, écart faible -> pas d'overfitting.
# - california_raw : val_loss plafonne à ~0.51 (≈ 2x le run normalisé). Adam évite
#   le NaN mais converge bien moins bien -> la normalisation fait la différence.
