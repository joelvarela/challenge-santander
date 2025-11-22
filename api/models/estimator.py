import joblib
import requests
from pathlib import Path

# 📌 Carpetas locales donde deben guardarse los modelos
ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "model" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# 📌 CDN's Modelos Entrenados
FINAL_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/final_model.joblib"
PIPELINE_MODEL_URL = "https://joelvarela.ar/cdn/artifacts/model_pipeline.joblib"

class Estimator:
    def __init__(self, use_pipeline=True):
        """
        use_pipeline = True  -> carga model_pipeline.joblib
        use_pipeline = False -> carga final_model.joblib
        """
        self.use_pipeline = use_pipeline

        # 📌 Definir rutas locales de los archivos
        self.final_model_path = ARTIFACTS / "final_model.joblib"
        self.pipeline_model_path = ARTIFACTS / "model_pipeline.joblib"

        # 📌 Descargar los archivos si no existen
        self._ensure_artifacts()

        # 📌 Cargar el archivo correcto
        self.pipeline = self._load_model()


    def _ensure_artifacts(self):
        """Descarga los modelos desde tu CDN si no están en el contenedor."""

        # --- Descargar final_model.joblib ---
        if not self.final_model_path.exists():
            print("📥 Descargando final_model.joblib desde CDN...")
            self._download(FINAL_MODEL_URL, self.final_model_path)

        # --- Descargar model_pipeline.joblib ---
        if not self.pipeline_model_path.exists():
            print("📥 Descargando model_pipeline.joblib desde CDN...")
            self._download(PIPELINE_MODEL_URL, self.pipeline_model_path)


    def _download(self, url, dst):
        """Descargar un archivo binario por streaming."""
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise RuntimeError(f"❌ Error descargando {url}")

        with open(dst, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✅ Archivo descargado en: {dst}")


    def _load_model(self):
        """Cargar el modelo correcto, pipeline o modelo final."""
        if self.use_pipeline:
            print("🧠 Cargando model_pipeline.joblib...")
            return joblib.load(self.pipeline_model_path)
        else:
            print("🧠 Cargando final_model.joblib...")
            return joblib.load(self.final_model_path)


    def predict(self, data):
        """Realizar predicciones con el modelo cargado."""
        return self.pipeline.predict(data).tolist()
