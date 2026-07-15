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
docker compose exec -d dl tensorboard --logdir=logs/fit --host 0.0.0.0
```

Puis ouvrir http://localhost:6006

## Datasets et phases

### California Housing (régression)
- [x] Phase 1 : pipeline chargement + normalisation (split avant scale, pas de fuite)
- [x] Phase 2 : baseline PMC régression (MAE test ≈ 0.35)
- [x] Phase 3 : diagnostic TensorBoard (normalisé vs brut)

### Pima Diabetes (classification binaire)
- [ ] Phase 4 : baseline
- [ ] Phase 5 : régularisation (L2 + early stopping)
- [ ] Phase 6 : recherche d'hyperparamètres (keras-tuner)

### Wine Quality (multiclasse)
- [ ] Phase 7 : baseline
- [ ] Phase 8 : BatchNorm

- [ ] Phase 9 : tableau comparatif cross-dataset + dataset personnel
