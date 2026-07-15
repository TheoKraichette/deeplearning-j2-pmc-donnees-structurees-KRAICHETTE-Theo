"""
Phase 1 - Pipeline California Housing (régression).
Ordre choisi : split PUIS scale, scaler fitté sur X_train seul.
Fitter sur X entier ferait fuiter les stats de val/test (data leakage).
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target

# test mis de côté (20 %), puis val = 20 % du reste
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

print(f"X_train shape : {X_train_norm.shape}")
print(f"X_val shape   : {X_val_norm.shape}")
print(f"X_test shape  : {X_test_norm.shape}")

print("\nX_train_norm mean :", np.round(X_train_norm.mean(axis=0), 4))
print("X_train_norm std  :", np.round(X_train_norm.std(axis=0), 4))
print("X_test_norm  mean :", np.round(X_test_norm.mean(axis=0), 4))  # proche de 0 mais non nul = pas de fuite

print("\nFeatures :", housing.feature_names)
print("Nombre de features :", len(housing.feature_names))
assert len(housing.feature_names) == 8

# cas limite : fitter sur X entier tire la moyenne du test vers 0 (fuite)
X_test_norm_leak = StandardScaler().fit(X).transform(X_test)
print("\n|mean| test, fit train  :", round(float(np.abs(X_test_norm.mean(axis=0)).mean()), 4))
print("|mean| test, fit X (fuite) :", round(float(np.abs(X_test_norm_leak.mean(axis=0)).mean()), 4))

# adversarial : le scaler ne borne rien, un outlier ressort délirant
X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
print("MedInc=99999 normalisé :", round(float(scaler.transform(X_extreme)[0, 0]), 1))
