"""
Phase 4 - Baseline PMC classification binaire Pima Diabetes.
Dataset déséquilibré (~65/35) : un modèle qui prédit toujours la classe majoritaire
ferait déjà ~65% d'accuracy sans rien apprendre. L'accuracy seule ne suffit donc pas ;
on vérifie que predict().mean() reste proche de 0.35 (le modèle voit les cas positifs).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)

print("Distribution des classes :")
print(df['Outcome'].value_counts())

# des 0 physiologiquement impossibles (Glucose, BMI, Insulin, SkinThickness) = NaN déguisés,
# laissés tels quels pour le baseline mais notés comme point de fragilité
print("\nZéros par colonne :")
print((df == 0).sum())

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

print(f"\nAccuracy d'un modèle naïf (toujours la classe majoritaire) : {np.bincount(y).max() / len(y):.3f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(8,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(X_train_norm, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=0)

print(f"\nval_accuracy finale (max) : {max(history.history['val_accuracy']):.4f}")
print(f"predict().mean() : {model.predict(X_test_norm, verbose=0).mean():.3f}  (proche de 0.35 = pas d'effondrement)")
