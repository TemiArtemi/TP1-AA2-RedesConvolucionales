# Registro de experimentos — Práctica 1 (AA2)

Clasificador de imágenes con CNN en PyTorch. Dataset: *Intel Image Classification* (6 categorías).

Todas las ejecuciones se realizan en **CPU** con la semilla fijada en **42**. La arquitectura base es la `ConvNet` de referencia del profesor (`example/clasificador-lengua-signos_colab.ipynb`), adaptada a 6 clases. Cada experimento se define como una `ExperimentConfig` en `src/config.py` y se ejecuta con `python -m src.run_experiment --config <ID>`; los artefactos (historial, curvas, matrices de confusión) se guardan en `reports/`.

---

## Metodología: por qué dos fases

La búsqueda manual se divide en **dos fases**. La separación no es arbitraria: responde a lo que midió E1.

**Fase 1 — encontrar el mejor preprocesado.** Cada experimento cambia **una sola cosa** respecto al anterior (augmentación, normalización, resolución) para aislar su efecto. Los cambios son de preprocesado, no de los hiperparámetros que enumera la guía, y se resuelven primero porque **condicionan todo lo demás**:

| Experimento | `best_epoch` | Épocas ejecutadas | Ventana productiva |
|---|:---:|:---:|---|
| E1 (sin pipeline) | 10 | 20 | 10 épocas |
| E2 (+ augmentación) | 22 | 32 | 22 épocas |
| E3 (+ normalización) | 34 | 44 | 34 épocas |
| E4 (+ resolución 128×128) | 31 | 41 | 31 épocas |

En E1 el modelo se satura en la época 10 y el early stopping corta en la 20. Si en ese punto hubiéramos ajustado épocas, paciencia o learning rate, los habríamos afinado contra un techo de cristal que la augmentación después elimina. **Los hiperparámetros hay que buscarlos sobre la configuración que se va a entregar**, y esa configuración incluye el preprocesado.

**Fase 2 — buscar los hiperparámetros del enunciado.** Fijado el preprocesado, se parte de la mejor base y se cambia **un hiperparámetro por experimento** hasta cubrir los siete de la guía. Ver «Fase 2 — Cribado de hiperparámetros» al final del documento.

### Elección de la base: E3

La Fase 1 dejó dos candidatas con resultados prácticamente idénticos:

| | E3 (96×96) | E4 (128×128) |
|---|---:|---:|
| Val accuracy | 0,8441 | 0,8575 |
| Test accuracy | **0,8925** | 0,8333 |
| Parámetros | **1.245.862** | 2.163.366 |

La diferencia en validación (1,34 pp) está dentro del margen de error de un conjunto de 372 imágenes, así que **ambas están empatadas**. Ante ese empate se elige **E3**: tiene menos parámetros (−42 %) y tarda menos por época, ya que procesa muchos menos píxeles. En test además sale por delante.

> La Fase 2 parte, por tanto, de **E3**.

---

## Resumen — Fase 1

| ID | Cambio respecto al anterior | Resolución | best_epoch | LR | batch | Val acc | **Test acc** |
|----|-----------------------------|:----------:|:----------:|:---:|:-----:|:-------:|:------------:|
| E1 | Baseline (arquitectura de referencia) | 96 | 10 | 0,001 | 32 | 0,7742 | 0,7527 |
| E2 | + Data Augmentation | 96 | 22 | 0,001 | 32 | 0,8306 | 0,7849 |
| **E3** | + Normalización | 96 | 34 | 0,001 | 32 | 0,8441 | **0,8925** |
| E4 | + Resolución 128×128 | 128 | 31 | 0,001 | 32 | 0,8575 | 0,8333 |

Objetivo: ≥ 0,85 en test. Con 186 imágenes el margen de error es de ±5 pp, así que las cifras de test deben leerse con cautela; **E3 lo alcanza con 0,8925** y E4 se queda en 0,8333.

> **Nota de reproducibilidad.** E1 se registró primero de forma ad-hoc (best_epoch 11, val 0,7688, test 0,7742). Al pasar todas las pruebas por el runner unificado (`src/run_experiment.py`) en el mismo entorno, E1 reproduce best_epoch 10, val 0,7742, test 0,7527. Las diferencias (~1-2 pp) proceden del no-determinismo de las reducciones multihilo en CPU, por lo que **todas las comparaciones de la tabla usan ejecuciones del mismo runner**. La sección de E1 conserva el análisis de la ejecución original.

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

---

## E2 — Data Augmentation

### Hipótesis / por qué este cambio

E1 concluyó que el límite no era la capacidad de la red sino la **escasez de datos**: en la época óptima apenas había brecha train/val, pero la red entraba pronto en sobreajuste. Se mantiene el conjunto curado (308 imágenes/clase) y se aplica **augmentación solo al conjunto de entrenamiento** para aumentar la diversidad efectiva sin nuevas imágenes. Val y test permanecen sin transformar, de modo que la comparación con E1 aísla el efecto de la augmentación (ni normalización ni cambios de arquitectura).

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Imágenes curadas | 308 por clase × 6 = 1.848 |
| Reparto | 70 / 20 / 10, estratificado, semilla 42 |
| Train / Val / Test | 1.290 / 372 / 186 |
| Resolución | 96×96 |
| Normalización | ninguna (`ToTensor` → [0, 1]) |
| Batch size | 32 |
| `num_workers` | 0 |

**Augmentación (solo train)**

| Transformación | Parámetro |
|---|---|
| `RandomHorizontalFlip` | p = 0,5 |
| `RandomVerticalFlip` | **desactivado** (en paisajes el eje vertical tiene significado: cielo/agua arriba, suelo abajo) |
| `RandomRotation` | ±10° |
| `ColorJitter` | brightness / contrast / saturation 0,3; hue 0,05 |
| `RandomAffine` | traslación 10 %, zoom 0,9–1,1 |

**Arquitectura — `ConvNet`** (idéntica a E1)

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
| `best_epoch` | **22** |
| `best_val_loss` | 0,5077 |
| `val_accuracy` en `best_epoch` | 0,8306 |
| Épocas ejecutadas | 32 (corte por early stopping) |
| Tiempo total | 531,8 s ≈ 8,9 min |
| **`test_accuracy`** | **0,7849** |
| `test_loss` | 0,5213 |
| Precision macro / weighted | 0,7916 / 0,7916 |
| Recall macro / weighted | 0,7849 / 0,7849 |
| F1 macro / weighted | 0,7790 / 0,7790 |

**Métricas por clase (test)**

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| buildings | 0,6053 | 0,7419 | 0,6667 | 31 |
| forest | 0,7778 | 0,9032 | 0,8358 | 31 |
| glacier | 0,9677 | 0,9677 | 0,9677 | 31 |
| mountain | 0,7778 | 0,4516 | 0,5714 | 31 |
| sea | 0,7879 | 0,8387 | 0,8125 | 31 |
| street | 0,8333 | 0,8065 | 0,8197 | 31 |

**Matriz de confusión** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           23         5         0         1         1         1
forest               0        28         0         0         0         3
glacier              0         0        30         1         0         0
mountain            10         1         0        14         6         0
sea                  1         0         1         2        26         1
street               4         2         0         0         0        25
```

### Observaciones

**La hipótesis se confirma: la augmentación retrasa el sobreajuste.** El mejor modelo pasa de la época 10 (E1) a la **22**, y el corte por early stopping de la 20 a la **32**. Al disponer de versiones perturbadas de cada imagen en cada época, la pérdida de entrenamiento avanza más despacio (en `best_epoch`, train loss 0,5794 frente a 0,5098 en E1) mientras la de validación seguiría bajando. Esto es exactamente el comportamiento que E1 predijo como consecuencia de tener más datos efectivos.

**Ganancia neta frente a E1 (mismo runner):** val +5,64 pp (0,7742 → 0,8306) y test +3,22 pp (0,7527 → 0,7849). El objetivo de 0,85 sigue sin alcanzarse (distancia ≈ 6,5 pp).

**La confusión se ha reorganizado hacia `mountain`.** `glacier` (F1 0,9677) y `forest` pasan a ser casi perfectos, y `buildings` recupera recall (0,7419 frente a 0,6452). Sin embargo, `mountain` sigue siendo la peor clase (recall 0,4516): 10 montañas se predicen como `buildings` y 6 como `sea`. Es el mismo núcleo semántico del problema (una montaña contiene edificios, agua y árboles), ahora más concentrado al haber mejorado el resto.

**Persiste una brecha val/test de ~4,6 pp** (0,8306 vs 0,7849). Esta pendiente indica que la validación no es del todo representativa del test y refuerza que el siguiente paso debe atacar la generalización, no solo el sobreajuste de entrenamiento.

### Siguiente paso previsto

Introducir **normalización** de entrada (`Normalize` con media/desvío por canal); da E3. La exploración de resolución, capacidad y learning rate se difiere a la Fase 2: primero hay que dejar cerrado el preprocesado, porque los hiperparámetros dependen de él — como muestra la tabla de `best_epoch` de la sección de metodología.

---

## E3 — Normalización

### Hipótesis / por qué este cambio

E2 mejoró la validación hasta 0,8306 pero dejó el test en 0,7849. Sin normalizar, los tres canales RGB entran en la red con escalas distintas (la media del canal rojo es 0,485 y la del azul 0,406), y la primera capa convolucional tiene que compensar esa diferencia por sí sola. Se añade `Normalize` con las medias y desviaciones estándar de ImageNet, aplicada a **los tres conjuntos**; la augmentación sigue siendo exclusiva de train.

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Imágenes curadas | 308 por clase × 6 = 1.848 |
| Reparto | 70 / 20 / 10, estratificado, semilla 42 |
| Train / Val / Test | 1.290 / 372 / 186 |
| Resolución | 96×96 |
| Normalización | `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` |
| Augmentación | **idéntica a E2**, solo en train |
| Batch size | 32 |
| `num_workers` | 0 |

**Arquitectura — `ConvNet`** (idéntica a E1/E2)

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
| `best_epoch` | **34** |
| `best_val_loss` | 0,4908 |
| `val_accuracy` en `best_epoch` | 0,8441 |
| Épocas ejecutadas | 44 (corte por early stopping) |
| **`test_accuracy`** | **0,8925** |
| `test_loss` | 0,3359 |
| Precision macro / weighted | 0,8926 / 0,8926 |
| Recall macro / weighted | 0,8925 / 0,8925 |
| F1 macro / weighted | 0,8918 / 0,8918 |

**Métricas por clase (test)**

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| buildings | 0,8387 | 0,8387 | 0,8387 | 31 |
| forest | 0,9677 | 0,9677 | 0,9677 | 31 |
| glacier | 1,0000 | 0,9677 | 0,9836 | 31 |
| mountain | 0,8214 | 0,7419 | 0,7796 | 31 |
| sea | 0,8529 | 0,9355 | 0,8923 | 31 |
| street | 0,8750 | 0,9032 | 0,8889 | 31 |

**Matriz de confusión — validación** (filas = clase real, columnas = predicha)

```text
             buildings    forest   glacier  mountain       sea    street
buildings           48         0         2         6         2         4
forest               2        59         0         0         0         1
glacier              0         0        56         2         4         0
mountain             5         0         3        44         8         2
sea                  0         0         1         4        55         2
street               6         1         0         2         1        52
```

**Matriz de confusión — test**

```text
             buildings    forest   glacier  mountain       sea    street
buildings           26         0         0         3         0         2
forest               0        30         0         0         0         1
glacier              0         0        30         0         1         0
mountain             3         1         0        23         4         0
sea                  1         0         0         0        29         1
street               1         0         0         2         0        28
```

### Observaciones

**Es el mejor test de la Fase 1 (0,8925) y la validación sigue subiendo.** Frente a E2: val +1,35 pp (0,8306 → 0,8441) y test +10,76 pp (0,7849 → 0,8925). Cuidado con la segunda cifra: con 186 imágenes el margen es de ±5 pp, así que hay que leerla como «E3 no ha empeorado el test y probablemente lo ha mejorado», no como una mejora de 10 puntos.

**`mountain` es la gran beneficiada.** Su recall pasa de 0,4516 (E2) a **0,7419** (E3): 9 montañas más clasificadas correctamente. Es el salto más grande de toda la Fase 1 por clase, y es justo la clase que E1 y E2 señalaban como límite del problema. Las 8 montañas que en E2 iban a `buildings` quedan reducidas a 3.

**`glacier` alcanza precisión perfecta** (1,0000): ningún ejemplo de otra clase se clasifica como glaciar. Su recall es 0,9677, con un único fallo hacia `sea`.

**La ventana de entrenamiento vuelve a alargarse:** `best_epoch` 22 → 34 y corte 32 → 44. La normalización hace que la optimización converja de forma más estable y que el modelo siga mejorando durante más épocas antes de saturar.

**La brecha val/test se invierte** (E2: val > test en 4,6 pp; E3: test > val en 4,8 pp). Este cambio de signo entre dos ejecuciones consecutivas es la mejor muestra de que las diferencias de test a esta escala son ruido.

### Siguiente paso previsto

Comprobar si la **resolución** aporta algo: E4 sube de 96×96 a 128×128 dejando todo lo demás idéntico.

---

## E4 — Resolución 128×128

### Hipótesis / por qué este cambio

Las imágenes originales del dataset miden 150×150. A 96×96 se descarta más de la mitad de los píxeles; a 128×128 se conserva bastante más detalle de bordes, texturas finas y líneas en perspectiva — útiles sobre todo para `street` y `buildings`. Se cambia **solo** `image_size`, dejando preprocesado, arquitectura y entrenamiento intactos, para aislar el efecto.

### Configuración

**Datos**

| Parámetro | Valor |
|---|---|
| Imágenes curadas | 308 por clase × 6 = 1.848 |
| Reparto | 70 / 20 / 10, estratificado, semilla 42 |
| Train / Val / Test | 1.290 / 372 / 186 |
| Resolución | **128×128** (E3: 96×96) |
| Normalización | idéntica a E3 |
| Augmentación | idéntica a E2/E3, solo en train |
| Batch size | 32 |
| `num_workers` | 0 |

**Arquitectura — `ConvNet`**

La red es la misma, pero al entrar imágenes mayores la rejilla antes del *flatten* pasa de 12×12 a 16×16, y con ella crece la capa `fc1`:

| Capa | Configuración | Parámetros |
|---|---|---:|
| `conv1` | Conv2d 3→16, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 1.216 |
| `conv2` | Conv2d 16→32, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 12.832 |
| `conv3` | Conv2d 32→64, k=5, s=1, p=2 + ReLU + MaxPool 2×2 | 51.264 |
| `fc1` | Linear **16384**→128 + ReLU + Dropout 0,25 | 2.097.280 |
| `fc2` | Linear 128→6 | 774 |
| | **Total** | **2.163.366** |

Las capas convolucionales no cambian en absoluto: **todos los parámetros nuevos (917.504) van a parar a `fc1`**, que pasa a concentrar el 97 % de la red.

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
| `best_epoch` | **31** |
| `best_val_loss` | 0,4611 |
| `val_accuracy` en `best_epoch` | 0,8575 |
| Épocas ejecutadas | 41 (corte por early stopping) |
| **`test_accuracy`** | **0,8333** |
| `test_loss` | 0,4625 |
| Precision macro / weighted | 0,8331 / 0,8331 |
| Recall macro / weighted | 0,8333 / 0,8333 |
| F1 macro / weighted | 0,8316 / 0,8316 |

**Métricas por clase (test)**

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| buildings | 0,7586 | 0,7097 | 0,7333 | 31 |
| forest | 0,8529 | 0,9355 | 0,8923 | 31 |
| glacier | 0,9677 | 0,9677 | 0,9677 | 31 |
| mountain | 0,8148 | 0,7097 | 0,7586 | 31 |
| sea | 0,8333 | 0,8065 | 0,8197 | 31 |
| street | 0,7714 | 0,8710 | 0,8182 | 31 |

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

**Matriz de confusión — test**

```text
             buildings    forest   glacier  mountain       sea    street
buildings           22         2         1         1         1         4
forest               0        29         0         0         0         2
glacier              0         0        30         1         0         0
mountain             3         1         0        22         4         1
sea                  2         0         0         3        25         1
street               2         2         0         0         0        27
```

### Observaciones

**Val y test se mueven en direcciones opuestas.** La validación sube 1,34 pp (0,8441 → 0,8575) y el test baja 5,92 pp (0,8925 → 0,8333). Cuando dos conjuntos discrepan así, lo más probable es ruido: 372 imágenes de validación tienen ±3,6 pp de margen y 186 de test, ±5 pp. **E3 y E4 están empatadas.**

**Las clases que supuestamente se beneficiarían tampoco se ponen de acuerdo.** En validación `street` gana 5 aciertos (52 → 57) y `mountain` 3 (44 → 47), que es donde encajaría la hipótesis de la resolución. Pero en test `buildings` pierde 4 (26 → 22) y `sea` pierde 4 (29 → 25), clases para las que el detalle fino no debería importar. Dos patrones sin relación: ruido.

**El coste es real y medible.** Procesar un 78 % más de píxeles por imagen y absorber 917.504 parámetros más en `fc1` encarece cada época y aumenta el riesgo de sobreajuste con los mismos 1.290 datos de entrenamiento.

**Conclusión de la Fase 1: E3 y E4 son intercambiables, y E3 es más barato.** Ver la sección de metodología para la elección de la base.

### Siguiente paso previsto

Cierre de la Fase 1 y apertura de la **Fase 2: búsqueda de hiperparámetros desde E3**, cubriendo los siete ejes que enumera la guía. Ver el plan al final del documento.

---

## Fase 2 — Cribado de hiperparámetros (E5–E12)

**8 configuraciones**, una por cada hiperparámetro de la lista del enunciado que faltaba. **Todas parten de E3** y cambian **una sola variable**, de modo que su efecto queda aislado. Todo lo no listado permanece igual que en E3: 96×96, normalización y augmentación, kernel 5, dropout 0,25, 128 neuronas FC, ReLU, LR 0,001, batch 32, techo 50 épocas.

| ID | Hiperparámetro | Cambio respecto a E3 | best_epoch | Val acc | Δ val | Test acc |
|----|----------------|----------------------|:----------:|--------:|------:|---------:|
| **E3** | *(referencia)* | — | 34 | **0,8441** | — | **0,8925** |
| E5 | Nº filtros por capa conv. | `conv_channels` 16-32-64 → **32-64-128** | 17 | 0,8199 | −2,42 | 0,8387 |
| E6 | Nº neuronas por capa FC | `fc_hidden_units` 128 → **256** | 30 | 0,8360 | −0,81 | 0,8602 |
| E7 | Tasa de aprendizaje | `learning_rate` 0,001 → **0,0005** | 28 | 0,8602 | +1,61 | 0,8656 |
| E8 | Tamaño de lote | `batch_size` 32 → **64** | 49 | 0,8602 | +1,61 | 0,8763 |
| E9 | Funciones de activación | `activation` ReLU → **ELU** | 35 | **0,8817** | **+3,76** | **0,9247** |
| E10 | Nº capas convolucionales | `conv_channels` 3 → **4** capas (16-32-64-128) | 31 | 0,8306 | −1,35 | 0,8065 |
| E11 | Nº capas FC | `fc_hidden_layer_count` 1 → **2** ocultas | 42 | 0,8522 | +0,81 | 0,8602 |
| E12 | Número de épocas | techo 50 → **100**, paciencia 10 → **15** | 34 | 0,8441 | 0,00 | 0,8925 |

**Lo destacado**

- **E9 (ELU) es el único resultado claro: +3,76 pp en validación y 0,9247 en test**, el mejor de los 12 en ambos conjuntos. Misma arquitectura y mismos 1.245.862 parámetros que E3, así que la mejora procede solo de la activación.
- **Más capacidad perjudica:** E5 (×2 filtros) −2,42 pp y E6 (×2 neuronas) −0,81 pp. Con 1.290 imágenes de *train*, añadir parámetros solo acrecenta el sobreajuste.
- **Más profundidad convolutiva tampoco ayuda:** E10 cae −1,35 pp en validación y −8,60 pp en test. Es la única red **más barata** (860.966 parámetros, −31 %), pero el cuarto *pooling* deja la rejilla en 6×6 y se pierde información espacial.
- **E12 devuelve exactamente E3:** misma `best_val_loss` (0,4908), mismo `best_epoch` (34), mismo test. El techo de épocas nunca llegó a mandar; quien decide es el *early stopping*. El hiperparámetro real es la **paciencia**, no el techo.
- E7 (+1,61), E8 (+1,61) y E11 (+0,81) rozan o caen dentro del ruido: **señal débil**, no ganancia probada.

**Cobertura de la lista del enunciado**

| Hiperparámetro | Fase 1 | Fase 2 |
|---|:---:|:---:|
| El número de épocas | — | E12 |
| La tasa de aprendizaje | — | E7 |
| El tamaño del lote | — | E8 |
| El número de capas convolucionales y FC | — | E10, E11 |
| El número de filtros por capa convolucional | — | E5 |
| El número de neuronas por capa FC | — | E6 |
| Funciones de activación | — | E9 |

Con E12 cerrada, la búsqueda queda en **12 configuraciones documentadas**, con los siete hiperparámetros cubiertos y cada uno aislado en su propio experimento. A ellas se suma **E13**, el ensamblado, que cierra la búsqueda al final del documento.

**Implementación.** E5–E10 y E12 fueron pura configuración en `src/config.py`. **E11 exigió modificar `src/model.py`**: `fc1`/`fc2` fijas → `nn.ModuleList` + `classifier` controlado por `architecture.fc_hidden_layer_count`. Verificado tras el cambio: E3 sigue construyendo exactamente **1.245.862** parámetros y las 12 redes hacen *forward* con salida `(batch, 6)`. E10 exige `image_size % 16 == 0`, que con 96 se cumple.

**Aviso de interpretación.** Se midió un no-determinismo de **~1-2 pp** entre ejecuciones del mismo modelo por las reducciones multihilo en CPU. Diferencias menores de esa magnitud **no deben leerse como mejoras reales**: es el motivo por el que E7, E8 y E11 quedan como señal débil y no como ganancia probada.

## E13 — Ensamblado (E9 + E7 + E8 + E12)

### Hipótesis / por qué este cambio

El cribado OFAT no detecta interacciones, así que E13 combina los cambios que ganaron más de 1,5 pp sobre E3: **E9 (ELU, +3,76) + E7 (LR 0,0005, +1,61) + E8 (lote 64, +1,61)**. E11 queda fuera (+0,81, bajo el umbral), igual que E5, E6 y E10, que empeoran. La pregunta concreta: **¿añade algo E7 + E8 encima de E9?**

E13 hereda además de E12 el **techo 100 / paciencia 15**. No era una cuarta variable experimental, sino un seguro: E8 había agotado las 50 épocas **sin que saltara el *early stopping*** (mejor en la 49), mientras que E12 probó lo contrario y salió idéntica a E3 *bit a bit*. Se sabía, por tanto, en qué condiciones actúa el techo, y se sabía que **es inerte** cuando el *early stopping* manda primero — así que incluirlo no podía perjudicar.

### Configuración

Idéntica a E3 salvo: `activation` **ELU** (E9), `learning_rate` **0,0005** (E7), `batch_size` **64** (E8), `num_epochs` **100** y `patience` **15** (E12). Mismos **1.245.862** parámetros que E3 y E9.

### Resultados

| Métrica | E9 | **E13** | E3 |
|---|---:|---:|---:|
| `best_epoch` | 35 | **29** | 34 |
| `best_val_loss` | 0,4456 | 0,4632 | 0,4908 |
| `val_accuracy` | **0,8817** | 0,8575 | 0,8441 |
| `test_accuracy` | **0,9247** | 0,8280 | 0,8925 |
| Épocas ejecutadas | 45 (corte) | 44 (corte) | 44 (corte) |

### Observaciones

**La regla decide: se queda E9.** E13 alcanza **0,8575** en validación, **−2,42 pp por debajo de E9** y por debajo de la barra de 0,90 que se fijó antes de ejecutar. En test la caída es mayor, **−9,67 pp** (0,9247 → 0,8280). Los dos conjuntos coinciden, así que no hay ambigüedad: **E7 y E8 no aportan nada encima de E9**.

**La predicción sobre el techo resultó ser errónea.** Se esperaba que combinar LR ×½ y lote ×½ ralentizara tanto el entrenamiento que E13 agotara las 100 épocas. En realidad **el *early stopping* saltó en la 44**, mucho antes del techo, y el mejor modelo quedó en la **29** — *antes* incluso que el 35 de E9. ELU acelera la convergencia lo bastante para compensar con creces la mitad de *learning rate*. El techo volvió a ser inerte: la paciencia 15 solo costó 5 épocas sobre la 10 y no mejoró el mejor modelo. Incluirlo fue, como se preveía, inocuo.

**Por qué perjudicó.** No se puede aislar con una sola ejecución, pero la lectura más parsimoniosa es la que ya anticipaban E7 y E8 aisladamente: **eran ruido favorable** — subían validación y bajaban test. Sumar ruido a una buena configuración solo la degrada. Queda descartada la hipótesis de interacción positiva entre ELU y un entrenamiento más lento.

**Cierre de la búsqueda de hiperparámetros: 13 configuraciones documentadas.** La ganadora es **E9** — val 0,8817, **test 0,9247**, mismos 1.245.862 parámetros y mismo coste por época que E3. Es el único experimento del trabajo que gana a E3 simultáneamente en validación **y** en test, y supera holgadamente el objetivo de ≥ 0,85.
