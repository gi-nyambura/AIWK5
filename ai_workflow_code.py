# AI Development Workflow Code for Assignment

# ================================
# PART 1: Predicting Student Dropout
# ================================

# --- Preprocessing ---
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

student_df = pd.read_csv("student_data.csv")
student_df.fillna(method='ffill', inplace=True)

scaler = StandardScaler()
student_df[['grades', 'attendance']] = scaler.fit_transform(student_df[['grades', 'attendance']])

le = LabelEncoder()
student_df['study_mode'] = le.fit_transform(student_df['study_mode'])

# --- Model Development ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = student_df.drop('dropout', axis=1)
y = student_df['dropout']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Evaluation ---
from sklearn.metrics import accuracy_score, recall_score

y_pred = model.predict(X_test)
print("Student Dropout Prediction:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# ================================
# PART 2: Predicting Hospital Readmission
# ================================

# --- Preprocessing ---
data = pd.read_csv("hospital_readmission.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data['prior_admissions'] = data['total_admissions'] - 1

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
diagnosis_encoded = ohe.fit_transform(data[['diagnosis_code']])

# --- Model Development ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

X = data.drop('readmitted', axis=1)
y = data['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Hospital Readmission Prediction:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# --- FastAPI Deployment ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class PatientData(BaseModel):
    age: int
    gender: str
    comorbidities: int
    length_of_stay: float

@app.post("/predict")
def predict(data: PatientData):
    input_data = [[data.age, data.comorbidities, data.length_of_stay]]
    prediction = model.predict(input_data)[0]
    return {"readmission_risk": bool(prediction)}

# To run: uvicorn filename:app --reload
