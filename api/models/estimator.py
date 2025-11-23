import joblib
import requests
import pandas as pd
from pathlib import Path

# 📁 Ruta local donde deben guardarse los modelos descargados
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# 🌐 URLs de tu CDN
FINAL_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/final_model.joblib"
PIPELINE_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/model_pipeline.joblib"

class Estimator:
    def __init__(self, use_pipeline=False):
        """
        use_pipeline = False → carga final_model.joblib (MODELO COMPLETO)
        use_pipeline = True  → carga model_pipeline.joblib (solo transformers) ❌ NO RECOMENDADO
        """
        self.use_pipeline = use_pipeline

        # Rutas locales
        self.final_model_path = ARTIFACTS / "final_model.joblib"
        self.pipeline_model_path = ARTIFACTS / "model_pipeline.joblib"

        # Descargar si no existen
        self._ensure_artifacts()

        # Cargar modelo final
        self.model = self._load_model()


    # --------------------------------------------------------
    # 🔽 Descarga de artefactos
    # --------------------------------------------------------
    def _ensure_artifacts(self):
        """Descarga los archivos desde tu CDN solo si no existen."""
        
        if not self.final_model_path.exists():
            print("📥 Descargando final_model.joblib desde CDN...")
            self._download(FINAL_MODEL_URL, self.final_model_path)

        if not self.pipeline_model_path.exists():
            print("📥 Descargando model_pipeline.joblib desde CDN...")
            self._download(PIPELINE_MODEL_URL, self.pipeline_model_path)


    def _download(self, url, dst):
        """Descarga un archivo binario por streaming."""
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise RuntimeError(f"❌ Error descargando: {url}")

        with open(dst, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Archivo descargado en: {dst}")


    # --------------------------------------------------------
    # 📦 Carga del modelo
    # --------------------------------------------------------
    def _load_model(self):
        """Cargar el modelo correcto (pipeline o modelo final)."""
        if self.use_pipeline:
            print("⚠️ CUIDADO: Cargando model_pipeline.joblib (NO tiene .predict())")
            return joblib.load(self.pipeline_model_path)
        else:
            print("🧠 Cargando final_model.joblib...")
            return joblib.load(self.final_model_path)


    # --------------------------------------------------------
    # 🔮 Predicción
    # --------------------------------------------------------
    def predict(self, data):
        """
        Recibe un dict o lista de dicts,
        los convierte en DataFrame y llama a predict().
        """
        # Convertir dict → lista de dicts
        if isinstance(data, dict):
            data = [data]

        # Convertir a DataFrame
        df = pd.DataFrame(data)

        # Predicción
        preds = self.model.predict(df)

        return preds.tolist()
