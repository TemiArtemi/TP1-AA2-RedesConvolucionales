## E5

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
| Canales convolucionales | 32 - 64 - 128 |
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
| best_epoch | 17 |
| best_val_loss | 0.5491 |
| épocas ejecutadas | 27 |
| val accuracy | 0.8199 |
| **test accuracy** | **0.8387** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.5491 |
| accuracy | 0.8199 |
| precision macro | 0.8221 |
| precision weighted | 0.8221 |
| recall macro | 0.8199 |
| recall weighted | 0.8199 |
| f1 macro | 0.8202 |
| f1 weighted | 0.8202 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           48         1         0         6         3         4
forest               2        58         0         0         0         2
glacier              1         0        55         3         3         0
mountain             2         0         1        51         7         1
sea                  1         0         6         9        45         1
street               7         3         0         1         3        48
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4309 |
| accuracy | 0.8387 |
| precision macro | 0.8464 |
| precision weighted | 0.8464 |
| recall macro | 0.8387 |
| recall weighted | 0.8387 |
| f1 macro | 0.8393 |
| f1 weighted | 0.8393 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           23         2         0         2         2         2
forest               0        29         0         1         0         1
glacier              0         0        29         2         0         0
mountain             2         1         0        27         1         0
sea                  0         0         0         6        24         1
street               4         3         0         0         0        24
```
