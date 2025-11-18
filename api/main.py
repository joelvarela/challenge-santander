from models import Estimator

app = fastapi.FastAPI()


@app.post("/predict", status_code=200)
async def post_predict(input: dict) -> dict:

    # Instanciación del modelo
    model = Estimator()
    print(model)

    # Predicción
    try:
        result = model.predict(data)
    except Exception:
        print("Error :(")

    return result