## E12

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
| Learning rate | 0.001 |
| Épocas (techo) | 100 |
| Paciencia | 15 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 34 |
| best_val_loss | 0.4908 |
| épocas ejecutadas | 49 |
| val accuracy | 0.8441 |
| **test accuracy** | **0.8925** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4908 |
| accuracy | 0.8441 |
| precision macro | 0.8450 |
| precision weighted | 0.8450 |
| recall macro | 0.8441 |
| recall weighted | 0.8441 |
| f1 macro | 0.8439 |
| f1 weighted | 0.8439 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           48         0         2         6         2         4
forest               2        59         0         0         0         1
glacier              0         0        56         2         4         0
mountain             5         0         3        44         8         2
sea                  0         0         1         4        55         2
street               6         1         0         2         1        52
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.3359 |
| accuracy | 0.8925 |
| precision macro | 0.8926 |
| precision weighted | 0.8926 |
| recall macro | 0.8925 |
| recall weighted | 0.8925 |
| f1 macro | 0.8918 |
| f1 weighted | 0.8918 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           26         0         0         3         0         2
forest               0        30         0         0         0         1
glacier              0         0        30         0         1         0
mountain             3         1         0        23         4         0
sea                  1         0         0         0        29         1
street               1         0         0         2         0        28
```
