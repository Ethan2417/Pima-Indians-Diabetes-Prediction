# Pima Indians Diabetes Prediction

## Project Overview
This project focuses on predicting whether a patient has diabetes based on diagnostic measurements using the Pima Indians Diabetes dataset.

## Objective
To build and evaluate machine learning models that can accurately predict diabetes outcomes.

## Dataset
- Name: Pima Indians Diabetes Dataset
- Source: Kaggle
- https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Records: 768
- Features: 8 medical attributes + 1 target variable

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook
## How to Run This Project

1. Clone the repository:
```bash
git clone https://github.com/Ethan2417/Pima-Indians-Diabetes-Prediction.git

##Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn

##Open the Jupyter Notebook:
jupyter notebook notebooks/pima_diabetes_analysis.ipynb


##Data Preprocessing
- Replaced medically invalid zero values with median imputation
- Checked for missing values
- Feature scaling where required

##Exploratory Data Analysis
- Feature distributions
- Correlation analysis
- Outlier detection

## Models Used
- Logistic Regression
- Linear Discriminant Analysis (LDA)

## Model Evaluation
- Accuracy
- ROC–AUC Score
- Confusion Matrix

## Results
Logistic Regression and LDA models were evaluated, and the best model was selected based on ROC–AUC performance.

##Key Learnings
- Importance of data preprocessing in medical datasets
- Model evaluation using ROC–AUC
- Comparing linear classification models

## Project Structure

Pima-Indians-Diabetes-Prediction/
├── data/
│   └── pima_diabetes.csv
├── notebooks/
│   └── pima_diabetes_analysis.ipynb
├── images/
│   └── (plots and visualizations)
└── README.md

