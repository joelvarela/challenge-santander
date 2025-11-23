from fastapi import FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
from api.models.estimator import Estimator  # <-- importa con la ruta correcta

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API funcionando!"}

# 📌 Crear la instancia del modelo solo una vez al iniciar la API
# use_pipeline=True -> carga model_pipeline.joblib
# use_pipeline=False -> carga final_model.joblib
estimator = Estimator(use_pipeline=False)

# ------------------------------
# Esquema de entrada
# ------------------------------
class PredictionInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str
    income_cat: int


# ------------------------------
# Endpoint de predicción
# ------------------------------
@app.post("/predict")
async def post_predict(input: PredictionInput):
    data_dict = input.dict()   # transforma el input en dict

    try:
        # El estimator ahora devuelve siempre algo que se puede convertir a lista
        result = estimator.predict([data_dict])  # <-- IMPORTANTE: envolver en lista si tu modelo espera batch
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
