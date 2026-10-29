# TP1 AA2 — Clasificación de imágenes con redes convolucionales

Clasificador de imágenes en **PyTorch** para el dataset *Intel Image Classification*: 6 categorías (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`). Objetivo de la práctica: **≥ 85 % de acierto en test**.

Resultado final: **E9** (red entrenada desde cero) alcanza **0,9247** en test y **E14** (transfer learning con ResNet18) **0,9785**. El registro completo de las 15 configuraciones está en [`experiments.md`](experiments.md).

---

## ⚠️ Antes de nada: las imágenes no van en el zip

El fichero adjunto a la entrega **contiene solo el código** (pesa poco, que era el objetivo). **No incluye las fotografías**, que son 1.848 archivos y pesan bastante.

Las imágenes están en el repositorio de GitHub:

**<https://github.com/TemiArtemi/TP1-AA2-RedesConvolucionales>**

Para **ejecutar** el trabajo hace falta la carpeta `data_master/` con esas imágenes (**26,9 MB**, 1.848 ficheros). Hay dos formas de conseguirla:

- **Descargar el repositorio completo desde GitHub** — botón *Code* → *Download ZIP*. Ese sí incluye las imágenes.
- O bien clonar el repo: `git clone https://github.com/TemiArtemi/TP1-AA2-RedesConvolucionales.git`

Si solo se ha recibido el zip de la entrega, **el código se puede leer y estudiar sin problema**, pero los comandos de este README fallarán al no encontrar `data_master/`.

---

## Requisitos

- **Python 3.11** (probado con 3.11.9)
- Ejecutar todo en **CPU**; no hace falta GPU.

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

`requirements.txt` fija versiones concretas e incluye el índice de CPU de PyTorch, así que no se descarga la versión con CUDA (~2 GB de más).

---

## Preparar los datos

La carpeta `data/` **no está incluida**; se genera a partir de `data_master/`:

```powershell
python -m src.build_splits
```

Reparto: **70 / 20 / 10**, estratificado por clase y con semilla fijada. Debe imprimir:

```text
class            train       val      test
------------------------------------------
buildings          215        62        31
forest             215        62        31
glacier            215        62        31
mountain           215        62        31
sea                215        62        31
street             215        62        31
------------------------------------------
total             1290       372       186
```

215 imágenes por clase en entrenamiento, 62 en validación y 31 en test. El comando **borra y regenera** `data/` entera cada vez que se ejecuta, así que se puede lanzar sin miedo a acumular duplicados.

---

## Ejecutar las pruebas

Desde la **raíz del repositorio**, una configuración por comando:

```powershell
python -m src.run_experiment --config E9
```

Se sustituye `E9` por el identificador que se quiera probar. Cada ejecución entrena el modelo, aplica *early stopping*, guarda los pesos mejores y escribe los resultados en `reports/`.

---

## Qué prueba cada configuración

Las 15 configuraciones están en `src/config.py`. Todas comparten reparto de datos, semilla 42 y bucle de entrenamiento.

### Fase 1 — preprocesado (E1–E4)

Cada una cambia **una sola cosa** respecto a la anterior, para aislar su efecto.

| ID | Qué prueba |
|----|------------|
| E1 | Baseline: sin augmentación, sin normalización, 96×96 |
| E2 | + Data augmentation (solo en entrenamiento) |
| E3 | + Normalización con las medias y desviaciones de ImageNet |
| E4 | + Resolución 128×128 |

### Fase 2 — hiperparámetros (E5–E12)

Parten de **E3** y cambian **un hiperparámetro por experimento**, cubriendo los siete que pide el enunciado.

| ID | Qué prueba |
|----|------------|
| E5 | Nº de filtros por capa conv: 16-32-64 → **32-64-128** |
| E6 | Nº de neuronas por capa FC: 128 → **256** |
| E7 | Tasa de aprendizaje: 0,001 → **0,0005** |
| E8 | Tamaño de lote: 32 → **64** |
| E9 | Función de activación: ReLU → **ELU** |
| E10 | Nº de capas convolucionales: 3 → **4** |
| E11 | Nº de capas FC ocultas: 1 → **2** |
| E12 | Nº de épocas: techo 50 → **100**, paciencia 10 → **15** |

### Ensamblado y transfer learning (E13–E15)

| ID | Qué prueba |
|----|------------|
| E13 | Ensamblado: E9 + E7 + E8 + E12 combinados |
| E14 | **Transfer learning**: ResNet18 preentrenada en ImageNet, backbone congelado, solo se entrena la cabeza (66.438 parámetros) |
| E15 | Igual que E14 pero **descongelando también `layer4`** (8.460.166 parámetros) |

E14 y E15 usan ResNet18 a **224×224**, su resolución nativa. Todo lo demás (activación ELU, lote 32, LR 0,001, *early stopping*, augmentación, semilla) es idéntico a E9, de modo que la comparación con E9 aisla el efecto del modelo.

---

## Resultados esperados

Para comprobar que una reproducción cuadra. Selección por **validación**; el conjunto de test tiene solo 186 imágenes, así que sus cifras se leen con cautela.

| ID | Configuración | `best_epoch` | Val acc | **Test acc** |
|----|---------------|:------------:|--------:|-------------:|
| E1 | Baseline | 10 | 0,7742 | 0,7527 |
| E2 | + Augmentación | 22 | 0,8306 | 0,7849 |
| E3 | + Normalización | 34 | 0,8441 | 0,8925 |
| E4 | + Resolución 128 | 31 | 0,8575 | 0,8333 |
| E5 | + Filtros 32-64-128 | 17 | 0,8199 | 0,8387 |
| E6 | + Neuronas FC 256 | 30 | 0,8360 | 0,8602 |
| E7 | + LR 0,0005 | 28 | 0,8602 | 0,8656 |
| E8 | + Lote 64 | 49 | 0,8602 | 0,8763 |
| **E9** | **+ ELU — mejor desde cero** | **35** | **0,8817** | **0,9247** |
| E10 | + 4 capas conv | 31 | 0,8306 | 0,8065 |
| E11 | + 2 capas FC ocultas | 42 | 0,8522 | 0,8602 |
| E12 | + Techo 100 / paciencia 15 | 34 | 0,8441 | 0,8925 |
| E13 | Ensamblado | 29 | 0,8575 | 0,8280 |
| **E14** | **Transfer learning — mejor del trabajo** | **21** | **0,9543** | **0,9785** |
| E15 | Transfer learning + `layer4` | 7 | 0,9946 | 0,9731 |

**Objetivo ≥ 0,85 en test: lo alcanzan E3, E9, E11, E12 y E14.**

Sobre E15: supera a E14 en validación (0,9946) pero no en test (0,9731, una imagen de diferencia). Con 8,46 M de parámetros entrenables, el *early stopping* —que selecciona el mejor epoch mirando `val_loss`— acaba sobreajustando la validación por selección. El análisis está en `experiments.md`.

---

## Qué genera cada ejecución

En `reports/`:

| Fichero | Contenido |
|---|---|
| `<ID>_history.json` | Pérdida y precisión por época |
| `<ID>_curves.png` | Gráficas de pérdida y precisión (train vs val) |
| `<ID>_val_confusion.png` | Matriz de confusión en validación |
| `<ID>_test_confusion.png` | Matriz de confusión en test |
| `<ID>_summary.md` | Configuración y métricas completas |

En `checkpoints/`: `<ID>_best.pt`, los pesos del mejor epoch. **Esta carpeta pesa bastante** —el modelo de E14 tiene 11,2 M de parámetros— y no va en el zip de la entrega; se regenera al ejecutar. Las 15 configuraciones acumulan unos **155 MB**.

---

## A tener en cuenta

- **Ejecutar siempre desde la raíz del repositorio**, o los comandos no encuentran `data/` ni `src/`.
- **Solo CPU y semilla 42.** Todas las ejecuciones usan `torch.device("cpu")`.
- **Las cifras pueden variar ligeramente** (~1-2 pp) respecto a la tabla de arriba. Es ruido por las reducciones multihilo de CPU, no un fallo. Es también el motivo por el que la selección se hace por validación y no por test.
- **E14 y E15 descargan los pesos de ImageNet (44,7 MB) en la primera ejecución**, desde `download.pytorch.org`. Es automático y se guardan en caché en `~/.cache/torch/hub/checkpoints/`, pero **hace falta conexión a internet esa primera vez**.
- **No interrumpir** las ejecuciones; entrenan entera hasta que dispara el *early stopping*.
- **El código no lleva comentarios**, por requisito del enunciado. La documentación está en este README y en `experiments.md`.

---

## Estructura del repositorio

```text
├── data_master/          # 1.848 imágenes originales (6 clases × 308)
├── data/                 # particiones train/val/test — se genera, no versionada
├── reports/              # resultados de las 15 configuraciones
├── checkpoints/          # pesos .pt — se genera, no versionado
├── src/                  # código del proyecto
├── experiments.md        # registro completo del experimento
└── requirements.txt
```

### Módulos de `src/`

| Módulo | Qué hace |
|---|---|
| `build_splits.py` | Particiona `data_master/` en `data/train`, `data/val` y `data/test` (70/20/10, semilla 42) |
| `config.py` | Define `DataConfig`, `ArchitectureConfig` y `TrainingConfig`, y las 15 `ExperimentConfig` del registro |
| `dataset.py` | Construye las transformaciones (redimensionado, augmentación, normalización) y los `DataLoader` |
| `model.py` | `ConvNet`: la red convolucional entrenada desde cero, con canales, profundidad y capas FC configurables |
| `transfer_learning.py` | Los dos constructores de ResNet18 preentrenada: solo cabeza (E14) y cabeza + `layer4` (E15) |
| `train.py` | Bucle de entrenamiento y validación, con `CrossEntropyLoss`, Adam y *early stopping* |
| `early_stopping.py` | Clase `EarlyStopping`: vigila `val_loss`, corta tras la paciencia y restaura los mejores pesos |
| `evaluate.py` | Métricas en test y validación: precisión, recall, F1 (macro y ponderado) y matriz de confusión |
| `plotting.py` | Gráficas de curvas y matrices de confusión |
| `run_experiment.py` | Punto de entrada: monta datos, modelo y entrenamiento, y escribe los artefactos de `reports/` |

---

## Contenido de la entrega

- **Código sin comentarios** — `src/` y `requirements.txt`.
- **Enlace al repositorio de GitHub** — donde están las imágenes y el proyecto completo.
- **Resumen del proceso** — el informe, [`informe/main.pdf`](informe/main.pdf), con las gráficas de pérdida y precisión y las matrices de confusión. El código fuente LaTeX está en [`informe/main.tex`](informe/main.tex).
- **Registro de experimentos** — [`experiments.md`](experiments.md), con el detalle de las 15 configuraciones y el análisis de cada una.
