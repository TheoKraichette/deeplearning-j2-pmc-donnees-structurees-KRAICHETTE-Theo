"""
Phase 5 - Régularisation Pima Diabetes.
Cible : battre la val_accuracy du baseline (~0.77) tout en réduisant l'écart train/val.
Leviers testés dans l'ordre : L2 (petits poids -> frontière lisse), puis Dropout
(robustesse), le tout borné par Early Stopping (couper avant l'overfitting).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

keras.utils.set_random_seed(42)

pima_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(pima_url, names=cols)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)


def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    model = keras.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(8,),
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation="relu",
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

configs = {
    "baseline": build_pima_regularized(l2_lambda=0.0, use_dropout=False),
    "L2 seul": build_pima_regularized(l2_lambda=0.01, use_dropout=False),
    "L2 + Dropout": build_pima_regularized(l2_lambda=0.01, use_dropout=True),
}

histories = {}
for name, model in configs.items():
    h = model.fit(X_train_norm, y_train, epochs=300, validation_split=0.2,
                  callbacks=[early_stopping], verbose=0)
    histories[name] = h
    print(f"{name:14s} : arret epoch {len(h.history['val_loss']):3d}  |  "
          f"max val_accuracy = {max(h.history['val_accuracy']):.4f}")

plt.figure()
for name, h in histories.items():
    line, = plt.plot(h.history["val_loss"], label=name)
    plt.axvline(len(h.history["val_loss"]) - 1, color=line.get_color(), linestyle="--", alpha=0.4)
plt.xlabel("epoch")
plt.ylabel("val_loss")
plt.legend()
plt.savefig("phase5_pima_3configs.png")
