import joblib
import requests
import pandas as pd
from pathlib import Path

# 📌 Carpetas locales donde deben guardarse los modelos
ARTIFACTS = (
    Path(__file__).resolve().parent.parent.parent
    / "model"
    / "artifacts"
)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# 📌 CDN URLs
FINAL_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/final_model.joblib"
PIPELINE_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/model_pipeline.joblib"


class Estimator:
    def __init__(self):
        """
        Carga:
        - model_pipeline.joblib → para transformar inputs (OneHotEncoder, Scaler…)
        - final_model.joblib → para predecir
        """
        # Paths locales
        self.final_model_path = ARTIFACTS / "final_model.joblib"
        self.pipeline_model_path = ARTIFACTS / "model_pipeline.joblib"

        # Descargar si no existen
        self._ensure_artifacts()

        # Cargar archivos
        print("🧠 Cargando pipeline de transformación...")
        self.pipeline = joblib.load(self.pipeline_model_path)

        print("🧠 Cargando modelo final...")
        self.model = joblib.load(self.final_model_path)

    # ---------------------------------------------------------
    # DESCARGA DE ARCHIVOS
    # ---------------------------------------------------------
    def _ensure_artifacts(self):
        """Descarga los modelos desde tu CDN si no están en el contenedor."""

        if not self.final_model_path.exists():
            print("📥 Descargando final_model.joblib...")
            self._download(FINAL_MODEL_URL, self.final_model_path)

        if not self.pipeline_model_path.exists():
            print("📥 Descargando model_pipeline.joblib...")
            self._download(PIPELINE_MODEL_URL, self.pipeline_model_path)

    def _download(self, url, dst):
        """Descargar archivo binario por streaming."""
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise RuntimeError(f"❌ Error descargando {url}")

        with open(dst, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Archivo descargado en: {dst}")

    # ---------------------------------------------------------
    # PREDICCIÓN
    # ---------------------------------------------------------
    def predict(self, data):
        """
        Aplica el pipeline de transformación y luego
        usa el modelo final para predecir.
        """

        # Normalizar input: dict → lista
        if isinstance(data, dict):
            data = [data]

        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # 1️⃣ Transformación (OneHotEncoder + Scaling + Imputación)
        X_transf = self.pipeline.transform(df)

        # 2️⃣ Predicción
        preds = self.model.predict(X_transf)

        return preds.tolist()
