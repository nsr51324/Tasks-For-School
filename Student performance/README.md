# Student Performance Prediction

A simple machine learning project that predicts a student's total exam score based on personal and study-related information.

## About the Project

The **Exam Score** in the dataset is calculated by adding the student's Math, Reading, and Writing scores. The project trains and compares regression models to predict this total using:

- Gender
- Parental education level
- Lunch type
- Test preparation course completion
- Study time per week
- Number of absences

## Open the Notebook in Google Colab

[Open the project notebook in Google Colab](https://colab.research.google.com/drive/155B-wyyVHZj3kseKuqwCH3HSPRmPn-aC?usp=sharing)

## Run the App Locally

Install the required packages:

```bash
pip install gradio joblib pandas scikit-learn==1.6.1
```

From the project directory, run:

```bash
python app.py
```

Open the Gradio link shown in the terminal, enter the requested information, and view the predicted score.

## Project Files

- `app.py`: The interface for entering student information and viewing predictions.
- `data/`: The dataset used in the project.
- `notebooks/Task_school.ipynb`: Data analysis and model training workflow.
- `Model & Scaler/`: The trained model and data scaler.
- `Doc_Student_Performance_Report.docx`: The project report.

> The prediction is an estimate and may differ from the student's actual score.
