"""
Phase 9 (Partie B) - Pipeline complet sur dataset personnel : Breast Cancer Wisconsin.
Binaire (malin/bénin), 569 exemples, 30 features. Même recette que Pima mais plus de
features -> plus d'occasions d'overfitter, donc L2 + Dropout + Early Stopping d'emblée.
"""

import datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

data = load_breast_cancer()
X, y = data.data, data.target
print("shape :", X.shape)
print("distribution des classes :", np.bincount(y), "(0=malin, 1=benin)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(30,), kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

log_dir = "logs/breast_cancer/" + datetime.datetime.now().strftime("%H%M%S")
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

history = model.fit(X_train_norm, y_train, epochs=200, validation_split=0.2,
                    callbacks=[early_stop, tb_callback], verbose=0)

test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
y_proba = model.predict(X_test_norm, verbose=0).ravel()
print(f"epochs reels (early stopping) : {len(history.history['val_loss'])}")
print(f"val_loss finale : {history.history['val_loss'][-1]:.4f}")
print(f"test accuracy : {test_acc:.4f}")
print(f"test AUC : {roc_auc_score(y_test, y_proba):.4f}")
