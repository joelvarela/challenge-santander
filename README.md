# 🏦 Challenge Santander — Solución Completa  
*Entrenamiento auditable · ML en producción (API + Batch) · Mejora continua · Caso NLP extra*

Este repositorio contiene la solución al challenge técnico, incluyendo modelo, API y mejoras de ingeniería de ML.

---

## 1. 🔹 Funcionalidades implementadas en la rama `refactor`

En esta rama se trabajó para lograr:

- [x] Modelo de estimación de precios de propiedades (`model/`)
- [x] API de inferencia en FastAPI (`api/`) con:
  - [x] Endpoint `/predict` para predicciones
  - [x] Dockerfile para despliegue
  - [x] Validación básica de inputs
- [x] Contenedor listo para ejecución local o despliegue
- [x] Estructura modular del proyecto (`model/` y `api/` separadas)
- [x] Scripts de entrenamiento e inferencia básicos (`train.py` y `inference.py`)
- [x] Repositorio organizado con `requirements.txt` y README inicial

> ✅ Estado: funcional para pruebas locales y despliegue de la API.

---
 
## 2. 🛠 Roadmap Técnico: Funcionalidades faltantes / próximas mejoras

El objetivo de este roadmap es mostrar los pasos para transformar la solución actual en una versión completa de producción.

- **Tracking de experimentos y versionado de modelos**
  - Integrar MLflow o alternativa para registro de métricas y modelos
  - Registrar datasets y parámetros de entrenamiento
- **Pipeline modular con configuración**
  - Configuración centralizada (`config.yaml` o Hydra)
  - Separación de rutas, hiperparámetros y tipos de modelos
- **Batch processing para grandes volúmenes**
  - Script para predecir ~4M clientes mensualmente desde S3
  - Procesamiento en chunks y escritura eficiente de outputs
- **Monitoreo de modelos y drift**
  - Métricas de estabilidad (PSI)
  - Alertas ante degradación o cambio en la distribución de datos
- **CI/CD y despliegue automatizado**
  - Tests unitarios y linting
  - Construcción de contenedores y despliegue automático
  - Shadow mode y rollback para modelos nuevos
- **Mejora de features (NLP)**
  - Identificación de consultas sobre créditos hipotecarios
  - Feature adicional para el modelo de propensión

> ⚡ Recomendación: implementar incrementalmente, empezando por tracking y configuración modular, luego batch, monitoreo y CI/CD, y finalmente NLP.


---

## 📁 Estructura del Repositorio

```bash
.
├── model/
│ ├── data/
│ │ └── housing.csv
│ ├── training/
│ │ ├── config.yaml
│ │ ├── train.py
│ │ ├── utils.py
│ │ └── registry/
│ │ ├── runs/
│ │ └── models/
│ └── inference.py
│
├── api/
│ ├── main.py
│ ├── schemas.py
│ ├── service.py
│ ├── requirements.txt
│ └── Dockerfile
│
└── README.md
```

---

# 1. 🧪 Entrenamiento Reproducible, Auditable y Comparable

## 🎯 Objetivo
Convertir el script original de entrenamiento en un proceso:

- **Reproducible:** misma corrida = mismo resultado  
- **Auditable:** registro de métricas, parámetros, artefactos y dataset  
- **Comparable:** análisis entre corridas y entre modelos dentro de una corrida  

## 🛠️ Herramientas
- **MLflow:** tracking de experimentos  
- **Hydra:** gestión de configuraciones (config.yaml)  
- **Scikit-learn:** modelos predictivos  
- **Joblib:** serialización  
- **Pandas / Numpy**

## ✔️ Mejoras al pipeline de entrenamiento
- Registro automático de:
  - métricas  
  - hiperparámetros  
  - artefactos (modelo, scaler, dataset utilizado)  
- Control de semillas para reproducibilidad  
- Separación modular del código en `utils.py`  
- Posibilidad de agregar nuevos modelos fácilmente  
- Configuraciones externas sin tocar el código  

## 🚀 Cómo correr el entrenamiento

```bash
cd model/training
mlflow ui  # opcional
python train.py
```
Los resultados quedan almacenados en registry/runs/ y los modelos en registry/models/.

---

# 2. 🚀 API de Inferencia (FastAPI)

La API permite realizar predicciones en tiempo real durante la visita del tasador.

✔️ Mejoras realizadas

- Corrección de imports y errores de ejecución

- Validación de entrada con Pydantic

- Control horario (09:00–18:00)


## ▶️ Ejecutar la API

cd api
uvicorn main:app --host 0.0.0.0 --port 8000


## 📌 Ejemplo de request
- Logging estructurado

- Manejo de excepciones

- Endpoint /health

- Swagger en /docs

- Dockerfile para despliegue en producción

Swagger en /docs

Dockerfile para despliegue en producción

# 3. 🏗️ Arquitectura Batch Mensual (4M registros)
## 🎯 Objetivo

Ejecutar un modelo de propensión una vez por mes utilizando datos de ~4M clientes desde S3 y generar un archivo con predicciones también en S3.

## 📐 Arquitectura propuesta

       ┌───────────┐
       │   S3 Raw   │  (1.1GB CSV)
       └─────┬─────┘
             │
             ▼
     ┌───────────────┐
     │ AWS Batch /    │
     │ ECS Scheduled   │ ←— Trigger mensual (EventBridge)
     └──────┬────────┘
            │
            ▼
     ┌──────────────┐
     │ Contenedor ML │
     │  - Descarga data S3
     │  - Feature eng.
     │  - Carga modelo
     │  - Predicción (4M)
     │  - Output CSV
     └──────┬────────┘
            │
            ▼
        ┌───────────┐
        │   S3 Out   │ (~0.1GB)
        └───────────┘

## ⚙️ Escalabilidad

Procesamiento en chunks

I/O en streaming

Alternativa: Spark / Dask

Autoescalado con ECS/Fargate

# 4. ❗ Escenario: baja performance después de 6 meses
## 📉 Problema

El área de seguros reporta que casi ningún contacto convierte → el modelo dejó de ser efectivo.

## 🔍 Posibles causas

Data drift

Concept drift

Cambios en comportamiento del cliente

Features desactualizados

## 🛡️ ¿Cómo anticiparlo?

Monitoreo mensual o semanal

PSI (Population Stability Index)

Alertas automáticas

Seguimiento de tasa de conversión

## 🔧 ¿Cómo solucionarlo?

Reentrenamiento con datos recientes

Evaluación con métricas históricas

Ajuste del feature engineering

Validación previa al deployment del modelo

# 5. 🔁 CI/CD y automatización recomendada

Se automatizan:

Testing

Construcción del contenedor

Ejecución del pipeline de entrenamiento

Validación de métricas vs. modelo anterior

Promoción del modelo (Staging → Production)

Lanzamiento automático del job batch

# 6. 🛡️ Cómo asegurar que un modelo nuevo no sea peor
Estrategia utilizada:

Model Registry (MLflow)

Comparación de performance con el modelo actual

Shadow mode (predice en paralelo)

Rollback automático si empeora

# 7. 📞 Caso Extra — Clasificación de consultas sobre Créditos Hipotecarios

El CRM clasifica todo como “Consulta Créditos”, pero solo 1% es verdaderamente hipotecario.

✔️ Solución basada en NLP

```bash
Grabaciones + metadatos
        │
        ▼
 Speech-to-Text (Whisper / AWS Transcribe)
        │
        ▼
 Limpieza y normalización del texto
        │
        ▼
 Clasificador NLP binario:
    - Embeddings (BERT / DistilBERT)
    - Modelo supervisado (LR / SVM / LightGBM / Fine-tuned BERT)
        │
        ▼
 Etiquetas hipotecario (0/1)
        │
        ▼
 Feature incorporada al modelo de propensión
```
## Beneficios

Reduce ruido

Aumenta especificidad

Identifica realmente consultas hipotecarias

## 📝 Recomendaciones adicionales (opcional)

Tests unitarios (pytest)

Makefile para facilitar comandos

Hooks de calidad (black, flake8, isort)

Docker Compose para API + MLflow

Diagramas Mermaid

## 👤 Autor

Joel Varela
