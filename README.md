# Intelligent-Hybrid-Recommendation-System

Machine Learning Major Capstone Project

## Indian Car Market Analysis & Price Prediction

A Streamlit dashboard for:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Used Car Market Insights
- Machine Learning Price Prediction

---

## Project Structure

```text
Intelligent-Hybrid-Recommendation-System/
│
├── app.py
├── env.py
├── README.md
│
├── Cap_Training_Data_2025.csv
├── Cap_Test_Data_2025.csv
├── Cap_Sample_Submission_2025.csv
│
└── 4856969-Machine_Learning_Major_Project.pdf
```

---

---

## Dataset Files

Required files:

```text
Cap_Training_Data_2025.csv
Cap_Test_Data_2025.csv
Cap_Sample_Submission_2025.csv
```

Training file used by the dashboard:

```python
TRAINING_FILE = "Cap_Training_Data_2025.csv"
```

---

## Automatic Dataset Discovery

The application automatically searches for:

```text
Cap_Training_Data_2025.csv
```

---

## Environment Variables (Optional)

### Direct CSV Path

```powershell
$env:CAR_DATASET_PATH="C:\...\Cap_Training_Data_2025.csv"
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Machine Learning Models

The dashboard automatically evaluates:

- Ridge Regression
- Random Forest Regressor

The model with the lowest RMSE is selected automatically.

---

## Features

- Automatic Dataset Discovery
- Environment-Based Configuration
- Missing Value Handling
- Data Cleaning Pipeline
- Categorical Encoding
- Automated Model Selection
- Streamlit Dashboard
- Data Quality Analysis
- Market Analysis
- Price Prediction
- Model Performance Metrics

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Pathlib

---

## Author
Viswanadhu Sai Lohith 

** Sai Lohith**

Machine Learning Major Capstone Project
