import pandas as pd
import numpy as np
import joblib

# Load the final model and the full pipeline

# Pipeline Artifacts
columns = joblib.load("housing_columns.joblib")
num_pipeline = joblib.load("num_pipeline.joblib")
imputer = joblib.load("imputer.joblib")
ordinal_encoder = joblib.load("ordinal_encoder.joblib")
cat_encoder = joblib.load("cat_encoder.joblib")
full_pipeline = joblib.load("full_pipeline.joblib")

# Model
final_model = joblib.load("final_model.joblib")

data = pd.read_csv('data/example-data.csv', header=None)
data.columns = columns

data_transformed = full_pipeline.transform(data)
predictions = final_model.predict(data_transformed)
print(predictions)