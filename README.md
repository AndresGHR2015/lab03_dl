# Laboratorio 03 — Clasificación de Género y Regresión de Edad con UTKFace

**Autores:** Andrés Hidalgo Ramallo, Jeremy Araya Collao 
**Asignatura:** Deep Learning — Universidad Católica del Norte  
**Profesor:** Dr. Juan Bekios Calfa

---

## Descripción

Este proyecto implementa un sistema de **predicción multitarea** que, a partir de una única imagen facial del dataset [UTKFace (Aligned & Cropped)](https://susanqq.github.io/UTKFace/), realiza simultáneamente:

- **Clasificación discreta de género** (hombre / mujer) mediante `CrossEntropyLoss`.
- **Regresión continua de edad** (0–116 años) mediante `MSELoss`.

La función de pérdida combinada se define como:

$$\mathcal{L} = \mathcal{L}_{\text{género}} + \lambda \cdot \mathcal{L}_{\text{edad}}$$

donde $\lambda$ controla la contribución relativa de cada tarea.

---

## Arquitecturas Evaluadas

Se realizaron **6 experimentos** con complejidad creciente:

| ID  | Experimento                          | Descripción                                                         |
|-----|--------------------------------------|---------------------------------------------------------------------|
| E1  | Baseline Clásico                     | PCA (100 componentes) + GaussianNB (género) / Ridge (edad)          |
| E2  | MLP Multitarea                       | Red fully-connected con backbone compartido (512 → 128)             |
| E3  | CNN Simple Multitarea                | 3 bloques Conv2d + MaxPool con cabezas duales                       |
| E4  | ResNet18 (Transfer Learning)         | Backbone congelado con pesos ImageNet, solo se entrenan las cabezas |
| E5  | ResNet18 (Fine-Tuning)               | Backbone descongelado, se ajusta la red completa                    |
| E6  | Ablación (λ)                         | CNN con λ = 0.5 para analizar el impacto del balance de pérdidas    |

---

## Estructura del Proyecto

```
lab03_dl/
├── app.py                        # Interfaz Streamlit con pipeline de dos etapas
├── environment.yml               # Entorno Conda (agnóstico CUDA / Metal)
├── comparativa_modelos_utkface.png
├── data/
│   └── UTKFace/                  # Dataset (no versionado, ver instrucciones)
├── src/
│   ├── baseline.py               # E1: Pipeline clásico PCA + GaussianNB / Ridge
│   ├── dataset.py                # Carga, parseo y partición del dataset UTKFace
│   ├── models.py                 # Definición de MultiTaskMLP, MultiTaskCNN, MultiTaskResNet
│   ├── train.py                  # Loop de entrenamiento y validación con Early Saving
│   ├── eval.py                   # Evaluación con métricas por tarea y por década de edad
│   └── graphics.py               # Generación del gráfico comparativo de resultados
└── outputs/
    ├── models/                   # Checkpoints (.pth) y métricas (.json) por modelo
    ├── plots_and_csv/            # Ejemplos de predicciones correctas e incorrectas
    ├── baseline.log              # Log del baseline clásico
    └── entrenamiento.log         # Log del entrenamiento de redes neuronales
```

---

## Reproducibilidad

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/lab03_dl.git
cd lab03_dl
```

### 2. Crear el entorno Conda

El archivo `environment.yml` es agnóstico respecto al backend de aceleración. Conda resolverá automáticamente los paquetes de PyTorch apropiados según la plataforma (CUDA en servidor, Metal/CPU en macOS).

```bash
conda env create -f environment.yml
conda activate utkface_lab
```

### 3. Descargar el dataset

Descargar **UTKFace (Aligned & Cropped)** desde el [sitio oficial](https://susanqq.github.io/UTKFace/) y extraer las imágenes en:

```
data/UTKFace/
```

### 4. Semilla global

Toda la ejecución utiliza una **semilla determinista** (`seed = 42`) configurada en `src/dataset.py`, que fija los generadores de:

- `random` (Python)
- `numpy`
- `torch` (CPU y CUDA)

### 5. Hiperparámetros base

| Parámetro         | Valor              |
|-------------------|--------------------|
| Optimizador       | Adam               |
| Learning Rate     | 1e-3               |
| Batch Size        | 32                 |
| Épocas            | 20                 |
| λ (edad)          | 0.01 (0.5 en E6)  |
| Split Train/Val/Test | 70% / 15% / 15% |
| Workers           | 4                  |
| Resolución        | 224 × 224 px       |

---

## Ejecución del Entrenamiento

(Debido a su tamaño, los modelos pre-entrenados no están en el código fuente. Descarga el archivo models.zip desde la pestaña de Releases de este repositorio y colóca los modelos exactamente en la ruta outputs/models/ antes de ejecutar Streamlit. El programa esta hardcodeado para usar el modelo ResNet_FineTuned_best.pth por ser el mejor modelo. Esto se puede cambiar en app.y)

**Baseline clásico (E1):**

```bash
python -m src.baseline
```

**Modelos de Deep Learning (E2–E6):**

```bash
python -m src.train
```

Este comando entrena secuencialmente los 5 modelos restantes (MLP, CNN, ResNet congelada, ResNet fine-tuned, CNN ablación). Los checkpoints y métricas se almacenan en `outputs/models/`.

**Parámetros opcionales:**

```bash
python -m src.train --epochs 30 --lr 5e-4 --lambda-age 0.1 --data-dir data/UTKFace --output-dir outputs/models
```

**Gráfico comparativo:**

```bash
python src/graphics.py
```

---

## Despliegue con Streamlit

La aplicación de inferencia (`app.py`) constituye la **milla extra** del laboratorio. Se levanta con:

```bash
streamlit run app.py
```

### Pipeline de Dos Etapas

La inferencia en Streamlit implementa un **pipeline de preprocesamiento en dos etapas** para mitigar el *domain shift* entre fotografías reales y el dataset UTKFace:

1. **Detección y recorte facial (OpenCV):** Se utiliza el clasificador Haar Cascade (`haarcascade_frontalface_default.xml`) para localizar el rostro en la imagen de entrada. Se aplica un margen de 20 px y se recorta la región facial, eliminando fondos y contexto irrelevante.

2. **Inferencia multitarea (ResNet18 fine-tuned):** El rostro recortado se redimensiona a 224×224, se normaliza con las estadísticas de ImageNet y se pasa al modelo `MultiTaskResNet` (E5), que retorna simultáneamente la predicción de género y la edad estimada.

> **¿Por qué dos etapas?** El dataset UTKFace contiene rostros alineados y recortados. Pasar una fotografía completa (con cuerpo, fondo, múltiples personas) directamente al modelo introduce un *domain shift* significativo que degrada las predicciones. El recorte previo con Haar Cascades aproxima la distribución de entrada a la del entrenamiento.

---

## Métricas de Evaluación

- **Género:** Accuracy, Precision, Recall, F1-Score (ponderado), Matriz de Confusión.
- **Edad:** MAE (Error Absoluto Medio), RMSE, R², MAE desglosado por década de vida.

---

## Tecnologías

- **Python** 3.10
- **PyTorch** + TorchVision
- **scikit-learn** (baseline, métricas)
- **OpenCV** (detección facial)
- **Streamlit** (interfaz de inferencia)
- **Matplotlib** (visualización de resultados)
- **TensorBoard** (monitoreo del entrenamiento)
