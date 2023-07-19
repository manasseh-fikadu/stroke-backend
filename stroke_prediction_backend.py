import pickle
from fastapi import FastAPI, Query
# import sklearn
import xgboost

app = FastAPI()

# Load the trained model (Replace 'path_to_your_model' with the actual file path)
model = pickle.load(open('xgb-0.95roc.pkl', 'rb'))

@app.get('/')
def get_home():
    return {'message': 'API top level working'}

@app.get('/predict/')
def predict_stroke(
        gender: str = Query(..., description="Gender: 0 for female, 1 for male"),
        age: str = Query(..., description="Age of the person"),
        hypertension: str = Query(..., description="1 if the person has hypertension, 0 otherwise"),
        heart_disease: str = Query(..., description="1 if the person has heart disease, 0 otherwise"),
        ever_married: str = Query(..., description="1 if the person is ever married, 0 otherwise"),
        work_type: str = Query(..., description="Work type: 0 for private, 1 for self-employed, 2 for gov job, 3 for children, 4 for never worked"),
        Residence_type: str = Query(..., description="Residence type: 0 for rural, 1 for urban"),
        avg_glucose_level: str = Query(..., description="Average glucose level"),
        bmi: str = Query(..., description="Body mass index"),
        smoking_status: str = Query(..., description="Smoking status: 0 for unknown, 1 for never smoked, 2 for formerly smoked, 3 for smokes")
):
    # Prepare the input features as a list or array for the model prediction, cast all of them to thier respective types
    features = [int(gender), int(age), int(hypertension), int(heart_disease), int(ever_married), int(work_type), int(Residence_type), float(avg_glucose_level), float(bmi), int(smoking_status)]

    # Perform any necessary preprocessing if required (scaling, encoding, etc.)

    # Make predictions using the loaded model
    prediction = model.predict([features])[0]

    # Return the prediction result
    return {"stroke_prediction": bool(prediction)}
