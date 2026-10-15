## E2

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 96×96 |
| Normalización | ninguna |
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
| Épocas (techo) | 50 |
| Paciencia | 10 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 22 |
| best_val_loss | 0.5077 |
| épocas ejecutadas | 32 |
| val accuracy | 0.8306 |
| **test accuracy** | **0.7849** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.5077 |
| accuracy | 0.8306 |
| precision macro | 0.8413 |
| precision weighted | 0.8413 |
| recall macro | 0.8306 |
| recall weighted | 0.8306 |
| f1 macro | 0.8303 |
| f1 weighted | 0.8303 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           53         3         1         2         0         3
forest               2        57         0         0         0         3
glacier              2         0        56         0         4         0
mountain             9         0         6        41         5         1
sea                  2         0        10         1        48         1
street               7         0         0         1         0        54
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.5213 |
| accuracy | 0.7849 |
| precision macro | 0.7916 |
| precision weighted | 0.7916 |
| recall macro | 0.7849 |
| recall weighted | 0.7849 |
| f1 macro | 0.7790 |
| f1 weighted | 0.7790 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           23         5         0         1         1         1
forest               0        28         0         0         0         3
glacier              0         0        30         1         0         0
mountain            10         1         0        14         6         0
sea                  1         0         1         2        26         1
street               4         2         0         0         0        25
```
