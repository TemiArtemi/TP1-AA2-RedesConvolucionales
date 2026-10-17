## E5

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 128×128 |
| Normalización | sí |
| Augmentación | AugmentationConfig(horizontal_flip_probability=0.5, vertical_flip_probability=0.0, rotation_degrees=10.0, brightness_jitter=0.3, contrast_jitter=0.3, saturation_jitter=0.3, hue_jitter=0.05, translation_fraction=0.1, scale_minimum=0.9, scale_maximum=1.1) |
| Batch size | 32 |
| Semilla | 42 |

**Arquitectura**

| Parámetro | Valor |
|---|---|
| Canales convolucionales | 32 - 64 - 128 |
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
| best_epoch | 33 |
| best_val_loss | 0.4910 |
| épocas ejecutadas | 43 |
| val accuracy | 0.8414 |
| **test accuracy** | **0.8495** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4910 |
| accuracy | 0.8414 |
| precision macro | 0.8453 |
| precision weighted | 0.8453 |
| recall macro | 0.8414 |
| recall weighted | 0.8414 |
| f1 macro | 0.8407 |
| f1 weighted | 0.8407 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           49         1         0         4         2         6
forest               2        55         0         0         0         5
glacier              1         0        52         1         7         1
mountain             6         2         4        44         4         2
sea                  0         0         2         3        55         2
street               3         0         1         0         0        58
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4335 |
| accuracy | 0.8495 |
| precision macro | 0.8603 |
| precision weighted | 0.8603 |
| recall macro | 0.8495 |
| recall weighted | 0.8495 |
| f1 macro | 0.8475 |
| f1 weighted | 0.8475 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           24         1         0         1         0         5
forest               1        28         0         0         0         2
glacier              0         0        31         0         0         0
mountain             5         1         0        19         5         1
sea                  1         0         0         0        29         1
street               3         1         0         0         0        27
```
