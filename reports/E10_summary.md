## E10

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
| Canales convolucionales | 16 - 32 - 64 - 128 |
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
| best_val_loss | 0.5067 |
| épocas ejecutadas | 41 |
| val accuracy | 0.8306 |
| **test accuracy** | **0.8065** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.5067 |
| accuracy | 0.8306 |
| precision macro | 0.8325 |
| precision weighted | 0.8325 |
| recall macro | 0.8306 |
| recall weighted | 0.8306 |
| f1 macro | 0.8294 |
| f1 weighted | 0.8294 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           43         3         2         6         2         6
forest               2        56         0         0         0         4
glacier              1         0        53         1         6         1
mountain             3         0         3        46         7         3
sea                  1         0         1         3        56         1
street               3         2         0         2         0        55
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.5185 |
| accuracy | 0.8065 |
| precision macro | 0.8084 |
| precision weighted | 0.8084 |
| recall macro | 0.8065 |
| recall weighted | 0.8065 |
| f1 macro | 0.8009 |
| f1 weighted | 0.8009 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           23         2         0         3         0         3
forest               0        30         0         0         0         1
glacier              0         0        29         0         2         0
mountain             5         0         0        15         8         3
sea                  2         0         0         1        27         1
street               3         1         0         1         0        26
```
