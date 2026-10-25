## E11

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
| Épocas (techo) | 50 |
| Paciencia | 10 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 42 |
| best_val_loss | 0.4882 |
| épocas ejecutadas | 50 |
| val accuracy | 0.8522 |
| **test accuracy** | **0.8602** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4882 |
| accuracy | 0.8522 |
| precision macro | 0.8592 |
| precision weighted | 0.8592 |
| recall macro | 0.8522 |
| recall weighted | 0.8522 |
| f1 macro | 0.8533 |
| f1 weighted | 0.8533 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           52         0         1         4         2         3
forest               6        53         0         0         0         3
glacier              1         0        57         3         1         0
mountain             6         0         2        50         2         2
sea                  1         0         7         4        49         1
street               5         0         0         0         1        56
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4378 |
| accuracy | 0.8602 |
| precision macro | 0.8725 |
| precision weighted | 0.8725 |
| recall macro | 0.8602 |
| recall weighted | 0.8602 |
| f1 macro | 0.8610 |
| f1 weighted | 0.8610 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           29         0         0         1         0         1
forest               1        28         0         0         0         2
glacier              0         0        30         0         1         0
mountain             6         0         0        21         3         1
sea                  1         0         0         2        27         1
street               5         1         0         0         0        25
```
