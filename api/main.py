from fastapi import FastAPI
from pydantic import BaseModel
from api.models.estimator import Estimator  # importa el Estimator correctamente

app = FastAPI(title="Housing Price Estimator API")

# Instancia del modelo **una sola vez** al iniciar la API
estimator = Estimator()

# Define el esquema de input
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

@app.post("/predict")
async def post_predict(input: PredictionInput):
    data_dict = input.dict()
    try:
        # El Estimator debe recibir un dict o dataframe según tu implementación
        result = estimator.predict(data_dict)
        # Convertimos a lista para poder devolver JSON
        output = result.tolist() if hasattr(result, "tolist") else result
        return {"prediction": output}
    except Exception as e:
        return {"error": str(e)}
