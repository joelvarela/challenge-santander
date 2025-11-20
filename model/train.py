import pandas as pd
import numpy as np
import joblib
from pathlib import Path  # manejar paths correctamente
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# definir paths VARIABLES y limpias
DATA_PATH = Path("model/data/housing.csv")
ARTIFACT_PATH = Path("model/artifacts")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # variable de estratificación reproducible
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    return df


def stratified_split(df):
    # usar stratified split para evitar sesgo
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df["income_cat"]):
        return df.loc[train_idx], df.loc[test_idx]


def preprocess(train_df):
    housing = train_df.drop("median_house_value", axis=1)
    labels = train_df["median_house_value"].copy()

    # extracción limpia de columnas
    numeric_cols = housing.drop("ocean_proximity", axis=1).columns.tolist()
    categorical_cols = ["ocean_proximity"]

    # pipeline único y consistente
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ])

    # devolver pipeline + features procesadas
    X_prepared = full_pipeline.fit_transform(housing)

    return X_prepared, labels, full_pipeline


def train_best_model(X, y):
    # hiperparámetros moderados (evitan demoras en Render)
    param_grid = {
        "n_estimators": [80, 120],
        "max_features": [6, 8],
        "bootstrap": [True]
    }

    model = RandomForestRegressor(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,                       # más rápido
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1
    )

    grid.fit(X, y)

    # solo devolvemos el mejor modelo
    best_model = grid.best_estimator_
    return best_model


if __name__ == "__main__":
    print("=== Loading data")
    df = load_data()

    print("=== Splitting")
    train_df, test_df = stratified_split(df)

    print("=== Preprocessing")
    X_train, y_train, pipeline = preprocess(train_df)

    print("=== Training best model")
    final_model = train_best_model(X_train, y_train)

    ARTIFACT_PATH.mkdir(exist_ok=True)

    print("=== Saving artifacts")
    # guardamos pipeline y modelo en carpeta limpia
    joblib.dump(pipeline, ARTIFACT_PATH / "model_pipeline.joblib")
    joblib.dump(final_model, ARTIFACT_PATH / "final_model.joblib")

    print("=== Evaluating")
    X_test = pipeline.transform(test_df.drop("median_house_value", axis=1))
    y_test = test_df["median_house_value"]
    rmse = np.sqrt(mean_squared_error(y_test, final_model.predict(X_test)))

    print(f"FINAL RMSE: {rmse:.2f}")
    print("=== Training finished successfully!")
