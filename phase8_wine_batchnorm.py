"""
Phase 8 - BatchNormalization sur Wine Quality (4 configs comparées sur TensorBoard).
Ordre adopté : Dense (linéaire) -> BatchNorm -> ReLU (recommandé Ioffe & Szegedy 2015).
Argument : normaliser AVANT l'activation donne à ReLU une distribution centrée-réduite
à chaque batch, ce qui accélère et stabilise la convergence. On teste aussi l'ordre
inverse (BN après ReLU) pour comparer.
"""

import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

keras.utils.set_random_seed(42)

wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df_wine = pd.read_csv(wine_url, sep=';')
df_wine['quality_3class'] = df_wine['quality'].apply(lambda q: 0 if q <= 4 else (1 if q <= 6 else 2))
X_wine = df_wine.drop(['quality', 'quality_3class'], axis=1).values
y_wine = df_wine['quality_3class'].values
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine)
scaler = StandardScaler()
X_wine_train = scaler.fit_transform(X_wine_train)
X_wine_test = scaler.transform(X_wine_test)


def build_wine_model(use_batchnorm=False, bn_before_activation=True, extra_layer=False):
    model = keras.Sequential()
    if use_batchnorm and bn_before_activation:
        model.add(layers.Dense(64, input_shape=(11,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
    else:
        model.add(layers.Dense(64, activation='relu', input_shape=(11,)))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    if extra_layer:
        model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

configs = {
    'sans_bn': lambda: build_wine_model(use_batchnorm=False),
    'bn_avant_activation': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=True),
    'bn_apres_activation': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=False),
    'bn_extra_couche': lambda: build_wine_model(use_batchnorm=True, bn_before_activation=True, extra_layer=True),
}

results = {}
for name, build_fn in configs.items():
    log_dir = f"logs/wine/{name}_{datetime.datetime.now().strftime('%H%M%S')}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    model = build_fn()
    history = model.fit(X_wine_train, y_wine_train, epochs=200, validation_split=0.2,
                        callbacks=[early_stop, tb_callback], verbose=0)
    results[name] = {'val_accuracy': max(history.history['val_accuracy']),
                     'epochs': len(history.history['val_loss'])}
    print(f"{name}: val_accuracy={results[name]['val_accuracy']:.4f}, stopped at epoch {results[name]['epochs']}")
