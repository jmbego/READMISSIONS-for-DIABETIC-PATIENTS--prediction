from flask import Flask, request, render_template
import pickle
import bz2
import pandas as pd

app = Flask(__name__)

# Load the bz2 compressed model
model_path = r"Deployment\app\ModelPickle\model.pkl.bz2"

def load_compressed_model(path):
    with bz2.BZ2File(path, "rb") as f:
        return pickle.load(f)

pipeline = load_compressed_model(model_path)

# Mapping renamed features back to original model features
rename_mapping = {
    "Encounter ID": "encounter_id",
    "Patient ID": "patient_nbr",
    "Inpatient visits": "number_inpatient",
    "Lab Procedures": "num_lab_procedures",
    "Primary labtest": "diag_1",
    "Secondary labtest": "diag_2",
    "Generic Medications": "num_medications",
    "Additional Diagnosis": "diag_3",
    "Discharge No": "discharge_disposition_id",
    "Hospital Time": "time_in_hospital",
    "Age": "age",
    "Diagnosis Total": "number_diagnoses"
}

# Reverse mapping for predictions
reverse_mapping = {0: 'No readmission', 1: 'more 30 days', 2: 'less 30 days'}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user input
        user_input = {rename_mapping[feature]: float(request.form.get(feature, 0)) for feature in rename_mapping}
        input_df = pd.DataFrame([user_input])  # Convert input to DataFrame

        # Ensure correct feature order
        input_df = input_df[pipeline.feature_names_in_]

        # Make prediction
        prediction = pipeline.predict(input_df)
        
        # Convert numerical prediction back to original format
        prediction_result = reverse_mapping.get(prediction[0], "Unknown")
        
        return render_template("index.html", prediction_text=f"Prediction Results: {prediction_result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    