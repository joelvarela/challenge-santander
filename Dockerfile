# 1️⃣ Imagen base ligera con Python 3.11
FROM python:3.11-slim

# 2️⃣ Definir el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3️⃣ Copiar el archivo de dependencias
COPY requirements.txt .

# 4️⃣ Actualizar pip e instalar dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5️⃣ Copiar todo el proyecto al contenedor
COPY . .

# 6️⃣ Crear carpeta de artefactos si no existe
RUN mkdir -p model/artifacts

# 7️⃣ Entrenar el modelo para generar artefactos
RUN python model/train.py

# 8️⃣ Exponer el puerto de la API
EXPOSE 8000

# 9️⃣ Comando por defecto para arrancar la API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
