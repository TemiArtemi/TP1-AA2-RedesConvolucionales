## E15

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
| Backbone | resnet18_layer4 |
| Cabeza | Linear(512 → 128) → elu → Dropout(0.5) → Linear(128 → 6) |
| Bloques entrenables | layer4 |
| Activación | elu |
| Parámetros totales | 11,242,950 |
| Parámetros entrenables | 8,460,166 |

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
| best_epoch | 7 |
| best_val_loss | 0.0121 |
| épocas ejecutadas | 17 |
| val accuracy | 0.9946 |
| **test accuracy** | **0.9731** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.0121 |
| accuracy | 0.9946 |
| precision macro | 0.9948 |
| precision weighted | 0.9948 |
| recall macro | 0.9946 |
| recall weighted | 0.9946 |
| f1 macro | 0.9946 |
| f1 weighted | 0.9946 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           62         0         0         0         0         0
forest               0        62         0         0         0         0
glacier              0         0        61         1         0         0
mountain             0         0         0        62         0         0
sea                  0         0         0         1        61         0
street               0         0         0         0         0        62
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.0606 |
| accuracy | 0.9731 |
| precision macro | 0.9735 |
| precision weighted | 0.9735 |
| recall macro | 0.9731 |
| recall weighted | 0.9731 |
| f1 macro | 0.9732 |
| f1 weighted | 0.9732 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           30         0         0         0         0         1
forest               0        31         0         0         0         0
glacier              0         0        30         0         1         0
mountain             0         0         0        30         1         0
sea                  0         0         0         1        30         0
street               1         0         0         0         0        30
```
