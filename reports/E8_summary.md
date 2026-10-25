## E8

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 96×96 |
| Normalización | sí |
| Augmentación | AugmentationConfig(horizontal_flip_probability=0.5, vertical_flip_probability=0.0, rotation_degrees=10.0, brightness_jitter=0.3, contrast_jitter=0.3, saturation_jitter=0.3, hue_jitter=0.05, translation_fraction=0.1, scale_minimum=0.9, scale_maximum=1.1) |
| Batch size | 64 |
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
| best_epoch | 49 |
| best_val_loss | 0.4446 |
| épocas ejecutadas | 50 |
| val accuracy | 0.8602 |
| **test accuracy** | **0.8763** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4446 |
| accuracy | 0.8602 |
| precision macro | 0.8648 |
| precision weighted | 0.8648 |
| recall macro | 0.8602 |
| recall weighted | 0.8602 |
| f1 macro | 0.8617 |
| f1 weighted | 0.8617 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           49         2         0         4         2         5
forest               2        58         0         0         0         2
glacier              4         0        54         2         2         0
mountain             6         0         1        52         2         1
sea                  1         1         2         2        56         0
street               8         1         0         1         1        51
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.3796 |
| accuracy | 0.8763 |
| precision macro | 0.8817 |
| precision weighted | 0.8817 |
| recall macro | 0.8763 |
| recall weighted | 0.8763 |
| f1 macro | 0.8767 |
| f1 weighted | 0.8767 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           25         2         0         2         0         2
forest               1        29         0         0         0         1
glacier              0         0        30         0         1         0
mountain             5         0         0        23         3         0
sea                  1         0         0         0        29         1
street               3         1         0         0         0        27
```
