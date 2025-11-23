from fastapi import FastAPI
from pydantic import BaseModel
from api.models.estimator import Estimator

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API funcionando!"}

# Crear instancia del Estimator sin argumentos
estimator = Estimator()

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
        result = estimator.predict([data_dict])
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
