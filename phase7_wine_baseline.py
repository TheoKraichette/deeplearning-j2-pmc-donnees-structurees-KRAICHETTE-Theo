"""
Phase 7 - Baseline PMC multiclasse Wine Quality (vin rouge).
Choix : classification multiclasse sur 3 classes agrégées (low/medium/high), softmax
+ sparse_categorical_crossentropy (labels entiers, pas de one-hot).
Ce qu'on perd : l'ORDRE des classes (low<medium<high traité comme 3 catégories
indépendantes). La régression garderait l'ordre mais imposerait un seuillage ; la
régression ordinale (qui encode l'ordre) est rare en Keras de base.
Dataset très déséquilibré (medium >80%) -> stratify au split, et on juge sur la
matrice de confusion, pas la seule accuracy (un modèle "toujours medium" fait ~82%).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')

print("Distribution des qualités brutes :")
print(df_wine['quality'].value_counts().sort_index())


def map_quality(q):
    if q <= 4:
        return 0  # low
    elif q <= 6:
        return 1  # medium
    return 2  # high


df_wine['quality_3class'] = df_wine['quality'].apply(map_quality)
print("\nDistribution agregee (3 classes) :")
print(df_wine['quality_3class'].value_counts().sort_index())

X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(11,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(3, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train_norm, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=0)

print(f"\nval_accuracy finale (max) : {max(history.history['val_accuracy']):.4f}")
print(f"accuracy naive (toujours 'medium') : {np.bincount(y_wine).max() / len(y_wine):.3f}")

y_pred = model.predict(X_test_norm, verbose=0).argmax(axis=1)
print("Matrice de confusion (test, lignes=vrai / colonnes=predit) :")
print(confusion_matrix(y_test, y_pred))
