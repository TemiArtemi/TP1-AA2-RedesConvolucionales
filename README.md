# TP1 AA2 — Clasificación de imágenes con CNN

## Puesta a punto (una sola vez)

1. Instalar dependencias:

   ```powershell
   pip install -r requirements.txt
   ```

2. Generar las particiones de datos desde `data_master/`:

   ```powershell
   python -m src.build_splits
   ```

   (La carpeta `data/` está en `.gitignore`, hay que regenerarla. Debe imprimir `total 1290 / 372 / 186`.)

## Ejecutar las pruebas

Desde la raíz del repositorio, una configuración por comando:

```powershell
python -m src.run_experiment --config E1
python -m src.run_experiment --config E2
python -m src.run_experiment --config E3
python -m src.run_experiment --config E4
python -m src.run_experiment --config E5
```

Las reservas **E6** y **E7** solo se ejecutan si E5 no llega al 85 % en test:

```powershell
python -m src.run_experiment --config E6
python -m src.run_experiment --config E7
```

| ID | Qué prueba |
|----|------------|
| E1 | Baseline (sin augmentación) |
| E2 | + Data augmentation |
| E3 | + Normalización |
| E4 | + Resolución 128×128 |
| E5 | + Más capacidad (filtros y neuronas FC) |
| E6 | + Learning rate 0,0005 (reserva) |
| E7 | + Activación ELU (reserva) |

## Qué genera cada prueba

En `reports/`:

- `<ID>_history.json` — pérdida y precisión por época.
- `<ID>_curves.png` — curvas de pérdida y precisión.
- `<ID>_val_confusion.png` y `<ID>_test_confusion.png` — matrices de confusión.
- `<ID>_summary.md` — resumen con la configuración y las métricas.

En `checkpoints/`: `<ID>_best.pt` (mejores pesos).

## A tener en cuenta

- Ejecutar siempre desde la raíz del repositorio.
- Solo CPU y semilla fija 42.
- Cada prueba puede tardar bastante (decenas de minutos según el equipo); no interrumpir.
- Al terminar, enviar la carpeta `reports/` completa para actualizar `experiments.md` y la memoria.
