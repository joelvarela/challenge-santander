import joblib

class Estimator():

    def __init__(self):

        self.pipeline = joblib.load("full_pipeline.joblib")
        self.model = joblib.load("final_model.joblib")

    def transform(self, data):
        data_transformed = self.full_pipeline.transform(data)
        return data_transformed

    def predict(self, data_transformed):
        
        prediction = self.model.predict(data_transformed) 

        return {
            "estimated_value": prediction
        }
