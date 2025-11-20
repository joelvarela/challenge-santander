import joblib
import pandas as pd
from pathlib import Path  # paths consistentes

# centralizamos path a artefactos
ARTIFACTS = Path("model/artifacts")


class Estimator:

    def __init__(self):
        # carga segura del pipeline
        self.pipeline = joblib.load(ARTIFACTS / "model_pipeline.joblib")

        # carga del modelo final
        self.model = joblib.load(ARTIFACTS / "final_model.joblib")

    def predict(self, payload: dict):
        # convertimos el JSON a DataFrame de 1 fila
        df = pd.DataFrame([payload])

        # procesamos con el pipeline original
        transformed = self.pipeline.transform(df)

        # predicción homogénea como float
        pred = self.model.predict(transformed)[0]
        return {"prediction": float(pred)}
