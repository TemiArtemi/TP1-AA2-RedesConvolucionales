## E14

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 224×224 |
| Normalización | sí |
| Augmentación | AugmentationConfig(horizontal_flip_probability=0.5, vertical_flip_probability=0.0, rotation_degrees=10.0, brightness_jitter=0.3, contrast_jitter=0.3, saturation_jitter=0.3, hue_jitter=0.05, translation_fraction=0.1, scale_minimum=0.9, scale_maximum=1.1) |
| Batch size | 32 |
| Semilla | 42 |

**Arquitectura**

| Parámetro | Valor |
|---|---|
| Backbone | resnet18_head_only |
| Cabeza | Linear(512 → 128) → elu → Dropout(0.5) → Linear(128 → 6) |
| Bloques entrenables | ninguno (backbone completo congelado) |
| Activación | elu |
| Parámetros totales | 11,242,950 |
| Parámetros entrenables | 66,438 |

**Entrenamiento**

| Parámetro | Valor |
|---|---|
| Learning rate | 0.001 |
| Épocas (techo) | 50 |
| Paciencia | 10 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 21 |
| best_val_loss | 0.1098 |
| épocas ejecutadas | 31 |
| val accuracy | 0.9543 |
| **test accuracy** | **0.9785** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.1098 |
| accuracy | 0.9543 |
| precision macro | 0.9553 |
| precision weighted | 0.9553 |
| recall macro | 0.9543 |
| recall weighted | 0.9543 |
| f1 macro | 0.9544 |
| f1 weighted | 0.9544 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           60         0         0         0         0         2
forest               0        62         0         0         0         0
glacier              0         0        55         7         0         0
mountain             1         0         3        58         0         0
sea                  0         0         0         1        60         1
street               2         0         0         0         0        60
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.0848 |
| accuracy | 0.9785 |
| precision macro | 0.9792 |
| precision weighted | 0.9792 |
| recall macro | 0.9785 |
| recall weighted | 0.9785 |
| f1 macro | 0.9785 |
| f1 weighted | 0.9785 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           31         0         0         0         0         0
forest               0        30         0         0         0         1
glacier              0         0        29         2         0         0
mountain             0         0         0        30         1         0
sea                  0         0         0         0        31         0
street               0         0         0         0         0        31
```
