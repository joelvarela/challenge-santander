from fastapi import FastAPI
from pydantic import BaseModel       # validación del input
from models.estimator import Estimator  # import explícito


# modelo de validación Pydantic
class HousingInput(BaseModel):
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


app = FastAPI()

# cargamos el modelo una sola vez (mejora performance)
estimator = Estimator()


@app.post("/predict")
def predict(input: HousingInput):
    # convertimos a dict para pasárselo al estimator
    return estimator.predict(input.dict())
