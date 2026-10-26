## E13

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
| Activación | elu |

**Entrenamiento**

| Parámetro | Valor |
|---|---|
| Learning rate | 0.0005 |
| Épocas (techo) | 100 |
| Paciencia | 15 |
| min_delta | 0.0 |
| Optimizador | Adam |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| best_epoch | 29 |
| best_val_loss | 0.4632 |
| épocas ejecutadas | 44 |
| val accuracy | 0.8575 |
| **test accuracy** | **0.8280** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4632 |
| accuracy | 0.8575 |
| precision macro | 0.8597 |
| precision weighted | 0.8597 |
| recall macro | 0.8575 |
| recall weighted | 0.8575 |
| f1 macro | 0.8576 |
| f1 weighted | 0.8576 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           53         0         1         2         2         4
forest               2        57         0         0         0         3
glacier              1         0        57         1         2         1
mountain             4         0         4        47         6         1
sea                  0         0         5         6        50         1
street               5         0         0         0         2        55
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.4154 |
| accuracy | 0.8280 |
| precision macro | 0.8361 |
| precision weighted | 0.8361 |
| recall macro | 0.8280 |
| recall weighted | 0.8280 |
| f1 macro | 0.8299 |
| f1 weighted | 0.8299 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           25         1         0         2         1         2
forest               2        26         0         0         0         3
glacier              0         0        29         1         0         1
mountain             5         1         1        22         2         0
sea                  1         0         1         3        26         0
street               5         0         0         0         0        26
```
