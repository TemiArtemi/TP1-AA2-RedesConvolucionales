## E6

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 96×96 |
| Normalización | sí |
| Augmentación | AugmentationConfig(horizontal_flip_probability=0.5, vertical_flip_probability=0.0, rotation_degrees=10.0, brightness_jitter=0.3, contrast_jitter=0.3, saturation_jitter=0.3, hue_jitter=0.05, translation_fraction=0.1, scale_minimum=0.9, scale_maximum=1.1) |
| Batch size | 32 |
| Semilla | 42 |

**Arquitectura**

| Parámetro | Valor |
|---|---|
| Canales convolucionales | 16 - 32 - 64 |
| Kernel | 5 |
| Neuronas FC | 256 |
| Dropout | 0.25 |
| Activación | relu |

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
| best_epoch | 30 |
| best_val_loss | 0.4831 |
| épocas ejecutadas | 40 |
| val accuracy | 0.8360 |
| **test accuracy** | **0.8602** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4831 |
| accuracy | 0.8360 |
| precision macro | 0.8422 |
| precision weighted | 0.8422 |
| recall macro | 0.8360 |
| recall weighted | 0.8360 |
| f1 macro | 0.8334 |
| f1 weighted | 0.8334 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           49         4         0         2         1         6
forest               1        58         0         0         0         3
glacier              1         0        59         0         1         1
mountain             5         0         7        45         2         3
sea                  0         0         9         9        42         2
street               3         0         1         0         0        58
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4716 |
| accuracy | 0.8602 |
| precision macro | 0.8628 |
| precision weighted | 0.8628 |
| recall macro | 0.8602 |
| recall weighted | 0.8602 |
| f1 macro | 0.8595 |
| f1 weighted | 0.8595 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           24         4         0         1         0         2
forest               1        30         0         0         0         0
glacier              0         0        30         1         0         0
mountain             4         1         0        22         4         0
sea                  2         0         0         2        26         1
street               2         1         0         0         0        28
```
