# Deep Learning J2 - PMC sur données structurées

Trois classifieurs Keras/TensorFlow sur trois datasets réels : régression, classification
binaire et multiclasse. Chaque dataset introduit une technique nouvelle (diagnostic
TensorBoard, régularisation, keras-tuner, BatchNorm).

## Lancer le projet

Tout tourne dans Docker.

```bash
docker compose up -d
docker compose exec dl python phase1_pipeline_california.py
```

TensorBoard :

```bash
docker compose exec -d dl tensorboard --logdir=logs --host 0.0.0.0
```

Puis ouvrir http://localhost:6006

## Datasets et phases

### California Housing (régression)
- [x] Phase 1 : pipeline chargement + normalisation (split avant scale, pas de fuite)
- [x] Phase 2 : baseline PMC régression (MAE test ≈ 0.38)
- [x] Phase 3 : diagnostic TensorBoard (normalisé vs brut)

### Pima Diabetes (classification binaire)
- [x] Phase 4 : baseline (val_accuracy ≈ 0.77)
- [x] Phase 5 : régularisation (L2 + Dropout + early stopping)
- [x] Phase 6 : keras-tuner (best ≈ 0.80, activation tanh)

### Wine Quality (multiclasse)
- [x] Phase 7 : baseline (3 classes agrégées, matrice de confusion)
- [x] Phase 8 : BatchNorm (4 configs, BN accélère la convergence)

### Synthèse
- [x] Phase 9 : tableau comparatif + pipeline Breast Cancer + post-mortem

## Tableau comparatif

| Dataset | Tâche | val_accuracy / val_MAE | val_loss finale | Epochs (best) | Meilleur lr | Meilleure taille couche | Régularisation |
|---|---|---|---|---|---|---|---|
| California Housing | Régression | MAE 0.38 | 0.29 (MSE) | 100 | 0.001 (défaut) | 64→32 | Aucune (baseline) |
| Pima Diabetes | Classification binaire | acc 0.77 (tuné ~0.80) | 0.52 (BCE) | 122 | 0.0005 (tuner) | 32→48 (tuner) | L2 + Early Stopping |
| Wine Quality | Multiclassification | acc 0.83 | 0.52 (CE) | 37 | 0.001 (défaut) | 64→32 + BN | L2 + BatchNorm |

Bonus (dataset perso, Partie B) : **Breast Cancer**, binaire, 30 features → test accuracy **0.965** / AUC **0.993**.

> **Lire les `val_loss`** : une loss de classification se compare à la loss d'un modèle *aléatoire* (`ln(2)=0.69` en binaire, `ln(3)=1.10` à 3 classes), pas à 0. Une loss ~0.50 avec ~80 % d'accuracy est donc saine : le modèle bat nettement le hasard sans être sur-confiant. Les `val_loss` ne se comparent pas entre lignes (MSE en régression, cross-entropy en classif).
> Chiffres pris à la **même epoch** (le meilleur modèle, `argmin val_loss`) pour que loss et accuracy décrivent le même modèle. **Seed fixé** (`keras.utils.set_random_seed(42)` en tête de chaque script) → résultats reproductibles d'un run à l'autre.

## Ce que le tableau révèle

**Hypothèses posées avant de remplir :**
- (a) Profite le plus de la régularisation : **Pima** — le plus petit (768 exemples) → le plus prompt à mémoriser.
- (b) En profite le moins : **California Housing** — 20 640 exemples, très dur à overfitter (laissé en baseline sans régul).
- (c) Facteur qui explique le plus l'écart : **la taille du dataset**.

**Confrontation :** confirmé. California tient tout seul ; Pima overfitte dès l'epoch 23 sans frein (early stopping), les configs régularisées tiennent jusqu'à ~93-120. Nuance mesurée : sur Pima la régularisation apporte de la **robustesse** (overfitting retardé ~4×), pas un meilleur score — le plafond ~0.77 est fixé par les données (petit, bruité, zéros-NaN), pas par le modèle. Le tuner (~0.80) fait un peu mieux avec `tanh` + `lr=0.0005`.

**Les 3 questions de réflexion :**
1. **Hyperparamètre le plus impactant** : le **learning rate et l'activation** (le tuner sort `tanh` dans 4 des 5 meilleurs trials, avec un `lr` bas). La taille des couches compte moins (meilleur trial : `units_1=32`). Cohérent avec le random search : quelques dimensions dominent (Bergstra & Bengio).
2. **Plus grand écart train/val** : **Pima**. Plus petit dataset + features bruitées (zéros-NaN) → mémorisation rapide (overfit dès l'epoch 23 en baseline).
3. **Early stopping le plus précoce** : à distinguer selon la cause. Le baseline Pima s'arrête tôt (epoch 23) parce qu'il **overfitte** vite (mauvais signe) ; Wine + BatchNorm s'arrête tôt (~38-48) parce qu'il **converge** vite (bon signe, BN accélère). Deux arrêts précoces, deux causes opposées.

## Post-mortem

1. **Ce qui m'a surpris** : que la régularisation ne fasse pas monter le score sur Pima (plafond 0.77) — j'attendais un gain, alors qu'elle apporte de la robustesse, pas de la précision. Et que 0.85 de val_accuracy sur Wine cache un modèle **quasi-aveugle à la classe « low »** (1/13 détecté, révélé par la matrice de confusion).
2. **Ce que je referais autrement** : fixer le seed dès le départ (`keras.utils.set_random_seed(42)`) — je l'ai ajouté après coup, ce qui m'a obligé à relancer tous les runs pour des chiffres reproductibles — et traiter les zéros-NaN de Pima (imputation médiane) plutôt que les laisser.
3. **Ce que je n'ai pas eu le temps d'explorer** : la régression ordinale pour Wine (qui garderait l'ordre low<medium<high), keras-tuner sur Wine et Breast Cancer, et `class_weight` pour forcer la détection de la classe minoritaire « low ».
