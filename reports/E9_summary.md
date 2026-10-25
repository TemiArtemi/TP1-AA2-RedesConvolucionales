## E9

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
| Activación | elu |

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
| best_epoch | 35 |
| best_val_loss | 0.4456 |
| épocas ejecutadas | 45 |
| val accuracy | 0.8817 |
| **test accuracy** | **0.9247** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.4456 |
| accuracy | 0.8817 |
| precision macro | 0.8864 |
| precision weighted | 0.8864 |
| recall macro | 0.8817 |
| recall weighted | 0.8817 |
| f1 macro | 0.8829 |
| f1 weighted | 0.8829 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           56         0         0         2         1         3
forest               4        56         0         0         0         2
glacier              1         0        55         1         4         1
mountain             2         0         0        54         3         3
sea                  0         0         2         5        54         1
street               6         0         0         2         1        53
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.2538 |
| accuracy | 0.9247 |
| precision macro | 0.9253 |
| precision weighted | 0.9253 |
| recall macro | 0.9247 |
| recall weighted | 0.9247 |
| f1 macro | 0.9247 |
| f1 weighted | 0.9247 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           29         0         0         1         0         1
forest               1        29         0         0         0         1
glacier              0         0        29         2         0         0
mountain             2         1         1        26         1         0
sea                  0         0         1         1        29         0
street               1         0         0         0         0        30
```
