# Registro de experimentos — Práctica 1 (AA2)

Clasificador de imágenes con CNN en PyTorch. Dataset: *Intel Image Classification* (6 categorías).

Todas las ejecuciones se realizan en **CPU** con la semilla fijada en **42**. La arquitectura base es la `ConvNet` de referencia del profesor (`example/p2_lenguaje_signos.ipynb`), adaptada a 6 clases.

---

## Resumen

| ID | Cambio respecto al anterior | best_epoch | LR | batch | Val acc | **Test acc** |
|----|-----------------------------|:----------:|:---:|:-----:|:-------:|:------------:|
| E1 | Baseline (arquitectura de referencia) | 11 | 0.001 | 32 | 0.7688 | **0.7742** |

Objetivo: ≥ 0,85 en test. Distancia actual: **−7,58 pp**.

---

## E1 — Baseline

### Hipótesis / por qué este cambio

Reproducir fielmente la arquitectura y el preprocesado del ejemplo del profesor para establecer un punto de partida medido. Sin augmentación, sin normalización, con la resolución (96×96) y el reparto de datos que ya se habían decidido.

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Imágenes curadas | 308 por clase × 6 = 1.848 |
| Reparto | 70 / 20 / 10, estratificado, semilla 42 |
| Train / Val / Test | 1.290 / 372 / 186 |
| Resolución | 96×96 |
| Normalización | ninguna (`ToTensor` → [0, 1]) |
| Augmentación | ninguna |
| Batch size | 32 |
| `num_workers` | 0 (evita bloqueos por *multiprocessing* en Windows) |

**Arquitectura — `ConvNet`**

| Capa | Configuración | Parámetros |
|---|---|---:|
| `conv1` | Conv2d 3→16, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 1.216 |
| `conv2` | Conv2d 16→32, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 12.832 |
| `conv3` | Conv2d 32→64, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 51.264 |
| `fc1` | Linear 9216→128 + ReLU + Dropout 0,25 | 1.179.776 |
| `fc2` | Linear 128→6 | 774 |
| | **Total** | **1.245.862** |

**Entrenamiento**

| Parámetro | Valor |
|---|---|
| Épocas (techo) | 50 |
| Early stopping | paciencia 10, `min_delta` 0,0 sobre `val_loss` |
| Optimizador | Adam, learning rate 0,001 |
| Función de activación | ReLU |
| Pérdida | CrossEntropyLoss |

### Resultados

| Métrica | Valor |
|---|---:|
| `best_epoch` | **11** |
| `best_val_loss` | 0,6245 |
| `val_accuracy` en `best_epoch` | 0,7688 |
| Épocas ejecutadas | 21 (corte por early stopping) |
| Tiempo total | 155,8 s ≈ 2,6 min |
| **`test_accuracy`** | **0,7742** |
| `test_loss` | 0,6077 |
| Precision macro / weighted | 0,7862 / 0,7862 |
| Recall macro / weighted | 0,7742 / 0,7742 |
| F1 macro / weighted | 0,7733 / 0,7733 |

Macro y weighted coinciden porque las 6 clases tienen exactamente 31 imágenes de test cada una; con soportes iguales ambas medias son equivalentes.

**Métricas por clase**

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| buildings | 0,5750 | 0,7419 | 0,6479 | 31 |
| forest | 0,7838 | 0,9355 | 0,8529 | 31 |
| glacier | 0,9667 | 0,9355 | 0,9508 | 31 |
| mountain | 0,6400 | 0,5161 | 0,5714 | 31 |
| sea | 0,8387 | 0,8387 | 0,8387 | 31 |
| street | 0,9130 | 0,6774 | 0,7778 | 31 |

**Matriz de confusión** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           23         4         0         3         0         1
forest               1        29         0         0         0         1
glacier              0         0        29         2         0         0
mountain             8         1         1        16         5         0
sea                  1         0         0         4        26         0
street               7         3         0         0         0        21
```

### Observaciones

**El sobreajuste es el factor limitante, y el early stopping lo contuvo.** La trayectoria de `val_loss` alcanzó su mínimo en la época 11 y a partir de ahí se desacopló del entrenamiento:

| Época | train_loss | val_loss | train_acc | val_acc |
|:-----:|-----------:|---------:|-----------:|--------:|
| 11 | 0,5078 | **0,6245** | 0,8008 | 0,7688 |
| 20 | 0,0975 | 1,0257 | 0,9659 | 0,7419 |
| 21 | 0,1266 | 1,0405 | 0,9543 | 0,7016 |

Entre la época 11 y la 21 la pérdida de entrenamiento cae un 75 % mientras la de validación sube un 67 %; la brecha de precisión llega a **22,4 pp** (0,9659 vs 0,7419). La red deja de generalizar y empieza a memorizar. El early stopping corta en la 21 y **restaura los pesos de la época 11**, que son los evaluados en test.

**El modelo no converge por capacidad, sino por escasez de datos.** En la época óptima (11) la brecha train/val era de solo 3,2 pp (0,8008 vs 0,7688): el sobreajuste aún no había golpeado. Lo que se dispara es la *pendiente*. Esto sugiere que con un conjunto de entrenamiento efectivamente mayor — augmentación — el entrenamiento podría avanzar más épocas antes de entrar en esa pendiente.

**La confusión se concentra en `mountain`, por motivos semánticos.** Es la peor clase (recall 0,5161): 8 montañas clasificadas como `buildings`, 5 como `sea`. `street` falla hacia `buildings` (7) y `forest` (3), y `buildings` hacia `forest` (4) y `mountain` (3). Una montaña contiene edificios, árboles y agua, por lo que se superpone con las otras tres categorías. Es un límite del problema más que del modelo.

Comparado con una prueba preliminar de 4 épocas, el bloque glacier/sea — que era el mayor foco de error (17 de 31 glaciares clasificados como `sea`) — **desaparece por completo**; el recall de `glacier` pasa de 0,4516 a 0,9355. El modelo sí aprende a separar hielo de agua con entrenamiento suficiente.

### Siguiente paso previsto

Introducir augmentación de datos en el conjunto de entrenamiento, dirigida a cerrar la brecha train/val identificada arriba.
