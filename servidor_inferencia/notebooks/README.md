# Notebooks

Area de trabajo para exploracion y pruebas del modelo.

Convencion actual:

- `01_autoencoder_initial_experiments.ipynb`: notebook original de experimentacion del autoencoder portado desde el proyecto MLOps anterior. Se conserva como borrador historico y justificacion inicial del modelo.
- `02_baseline_rms.ipynb`: baseline RMS/estadistico por ventanas.
- `03_vae_experiments.ipynb`: primeras pruebas VAE.
- `04_real_data_exploration.ipynb`: exploracion de capturas reales.
- `05_train_vae_real_data.ipynb`: entrenamiento base del VAE real. Se conserva como referencia historica.
- `06_model_evaluation_wandb.ipynb`: evaluacion reproducible de modelos contra etiquetas, baseline RMS, run controlada, figuras y W&B.
- `07_train_vae_experiments.ipynb`: entrenamiento de nuevas variantes VAE con W&B activado por defecto. Las variantes se editan en `../config/vae_experiments.json`.
- `08_train_supervised_slip_detector.ipynb`: entrenamiento de una rama supervisada CNN 1D para la anomalia sutil de slip/patinaje usando etiquetas manuales `real_slip_manual_001/002` y W&B.
- `09_train_slip_mil_experiments.ipynb`: entrenamiento final de la rama CNN MIL multiescala para slip, incluyendo validacion cruzada y registro de artefactos en W&B.

La logica reutilizable debe moverse a `training/` o `app/`; los notebooks no deben ser la unica fuente de implementacion.

## Modelo final en worker

El worker ya puede usar un detector hibrido con dos ramas:

- VAE: deteccion no supervisada de anomalia general, especialmente saltos/impactos.
- Slip CNN: deteccion supervisada de patinaje/frenazo sutil usando las etiquetas manuales.

Configuracion recomendada para pruebas de circuito mixto:

```bash
DETECTOR_TYPE=hybrid
MODEL_PATH=models/vae_real_v6_window1s_derived.pth
SLIP_MODEL_PATH=models/slip_mil_w30_50_100_testperf_plus_validation_v2.pth
SLIP_THRESHOLD=0.30
WINDOW_SIZE=100
WINDOW_STEP=25
TORCH_DEVICE=auto
```

`SLIP_THRESHOLD=0.30` es el umbral operativo final usado para mejorar recall de slip en live. El artefacto tambien conserva su umbral interno de entrenamiento, pero para demo y validacion final se usa este override.
