## E1

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Clases | 6 (buildings, forest, glacier, mountain, sea, street) |
| Resolución | 96×96 |
| Normalización | ninguna |
| Augmentación | ninguna |
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
| best_epoch | 10 |
| best_val_loss | 0.6145 |
| épocas ejecutadas | 20 |
| val accuracy | 0.7742 |
| **test accuracy** | **0.7527** |

### Resultados — validación

| Métrica | Valor |
|---|---:|
| loss | 0.6145 |
| accuracy | 0.7742 |
| precision macro | 0.7795 |
| precision weighted | 0.7795 |
| recall macro | 0.7742 |
| recall weighted | 0.7742 |
| f1 macro | 0.7750 |
| f1 weighted | 0.7750 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           45         0         2         9         2         4
forest               3        51         0         1         1         6
glacier              1         0        52         1         8         0
mountain             3         0         4        44        10         1
sea                  0         0        11         6        42         3
street               5         2         0         0         1        54
```

### Resultados — test

| Métrica | Valor |
|---|---:|
| loss | 0.6118 |
| accuracy | 0.7527 |
| precision macro | 0.7598 |
| precision weighted | 0.7598 |
| recall macro | 0.7527 |
| recall weighted | 0.7527 |
| f1 macro | 0.7536 |
| f1 weighted | 0.7536 |

**Matriz de confusión — test** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           20         0         0         6         1         4
forest               2        24         0         0         0         5
glacier              0         0        29         0         2         0
mountain             5         0         2        18         6         0
sea                  0         0         1         4        26         0
street               7         1         0         0         0        23
```
