# Base image con Python 3.10
FROM python:3.10-slim

# Evitar prompts de apt
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y código
COPY requirements.txt .
COPY model ./model
COPY api ./api

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Crear carpeta de artefactos (donde se guardarán los modelos)
RUN mkdir -p model/artifacts

# Entrenar el modelo y generar los artefactos
RUN python model/train.py

# Exponer puerto de la API
EXPOSE 8000

# Comando para correr la API con uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
