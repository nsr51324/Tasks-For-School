
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "Model & Scaler"
model = joblib.load(model_dir / "best_model.pkl")
scaler = joblib.load(model_dir / "scaler.pkl")

def predict(gender, education, lunch, preparation, study_time, absences):

    data = pd.DataFrame([[
        0 if gender == "Male" else 1,
        ["High School", "College", "Bachelor", "Master"].index(education),
        0 if lunch == "Free/Reduced" else 1,
        0 if preparation == "Uncompleted" else 1,
        study_time,
        absences
    ]], columns=[
        "Gender",
        "Parental Education Level",
        "Lunch Type",
        "Test Preparation Course",
        "Study Time",
        "Absences"
    ])

    data[["Study Time", "Absences"]] = scaler.transform(
        data[["Study Time", "Absences"]]
    )

    prediction = model.predict(data)

    return round(float(prediction[0]), 2)

app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(
            ["High School", "College", "Bachelor", "Master"],
            label="Parental Education Level"
        ),
        gr.Dropdown(
            ["Standard", "Free/Reduced"],
            label="Lunch Type"
        ),
        gr.Dropdown(
            ["Completed", "Uncompleted"],
            label="Test Preparation Course"
        ),
        gr.Number(label="Study Time (Hours per Week)", minimum=0),
        gr.Number(label="Absences", minimum=0)
    ],
    outputs=gr.Number(label="Predicted Exam Score"),
    title="Student Exam Performance Prediction",
    description="Predict student exam score using Machine Learning."
)

app.launch()
