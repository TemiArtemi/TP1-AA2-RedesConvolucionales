## E4

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
| best_epoch | 31 |
| best_val_loss | 0.4611 |
| épocas ejecutadas | 41 |
| val accuracy | 0.8575 |
| **test accuracy** | **0.8333** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4611 |
| accuracy | 0.8575 |
| precision macro | 0.8582 |
| precision weighted | 0.8582 |
| recall macro | 0.8575 |
| recall weighted | 0.8575 |
| f1 macro | 0.8569 |
| f1 weighted | 0.8569 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           49         2         1         3         1         6
forest               2        58         0         0         0         2
glacier              1         0        55         2         4         0
mountain             6         0         5        47         1         3
sea                  0         1         4         3        53         1
street               3         0         0         1         1        57
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4625 |
| accuracy | 0.8333 |
| precision macro | 0.8331 |
| precision weighted | 0.8331 |
| recall macro | 0.8333 |
| recall weighted | 0.8333 |
| f1 macro | 0.8316 |
| f1 weighted | 0.8316 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           22         2         1         1         1         4
forest               0        29         0         0         0         2
glacier              0         0        30         1         0         0
mountain             3         1         0        22         4         1
sea                  2         0         0         3        25         1
street               2         2         0         0         0        27
```
