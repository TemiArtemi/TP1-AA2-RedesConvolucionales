## E7

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
| Neuronas FC | 128 |
| Dropout | 0.25 |
| Activación | relu |

**Entrenamiento**

| Parámetro | Valor |
|---|---|
| Learning rate | 0.0005 |
| Épocas (techo) | 50 |
| Paciencia | 10 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 28 |
| best_val_loss | 0.4506 |
| épocas ejecutadas | 38 |
| val accuracy | 0.8602 |
| **test accuracy** | **0.8656** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4506 |
| accuracy | 0.8602 |
| precision macro | 0.8605 |
| precision weighted | 0.8605 |
| recall macro | 0.8602 |
| recall weighted | 0.8602 |
| f1 macro | 0.8595 |
| f1 weighted | 0.8595 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           48         3         1         4         3         3
forest               2        58         0         0         0         2
glacier              1         0        56         1         3         1
mountain             4         0         3        48         5         2
sea                  1         0         4         3        53         1
street               2         0         2         0         1        57
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4217 |
| accuracy | 0.8656 |
| precision macro | 0.8627 |
| precision weighted | 0.8627 |
| recall macro | 0.8656 |
| recall weighted | 0.8656 |
| f1 macro | 0.8629 |
| f1 weighted | 0.8629 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           20         3         0         5         0         3
forest               0        31         0         0         0         0
glacier              0         0        30         0         1         0
mountain             4         0         0        24         3         0
sea                  1         0         0         1        28         1
street               2         1         0         0         0        28
```
