"""
Phase 6 - Recherche d'hyperparamètres Pima avec keras-tuner (RandomSearch, 15 trials).
Protection contre le piège du tuner (si max_epochs est trop bas, il choisit l'archi
qui converge vite plutôt que la meilleure) : chaque essai tourne jusqu'à 100 epochs
avec EarlyStopping(patience=10) sur val_loss, assez pour départager. Le meilleur est
ensuite réentraîné jusqu'à 200 epochs.
"""

import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

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


def build_pima_model(hp):
    units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    activation = hp.Choice('activation', values=['relu', 'tanh'])
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    model = keras.Sequential()
    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


tuner = kt.RandomSearch(
    build_pima_model,
    objective='val_accuracy',
    max_trials=15,
    seed=42,
    directory='tuning_pima',
    project_name='pima_random',
)
tuner.search_space_summary()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tuner.search(X_train_norm, y_train, epochs=100, validation_split=0.2, callbacks=[early_stop], verbose=0)

best_hp = tuner.get_best_hyperparameters()[0]
print("\nMeilleurs hyperparamètres :")
for k in ['learning_rate', 'units_1', 'units_2', 'activation', 'dropout_rate']:
    print(f"  {k} = {best_hp.get(k)}")

tuner.results_summary(num_trials=5)

print("\nConfigs des 5 meilleurs trials (chercher les invariants) :")
for hp in tuner.get_best_hyperparameters(num_trials=5):
    print(" ", hp.values)

best_model = tuner.hypermodel.build(best_hp)
history_best = best_model.fit(X_train_norm, y_train, epochs=200, validation_split=0.2,
                              callbacks=[early_stop], verbose=0)
print(f"\nBest model val_accuracy : {max(history_best.history['val_accuracy']):.4f}")
