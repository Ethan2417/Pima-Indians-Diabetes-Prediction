# Pima Indians Diabetes Prediction

A machine learning classification project that predicts whether a patient is likely to have diabetes based on diagnostic and demographic features from the Pima Indians Diabetes dataset.

The project focuses on data exploration, preprocessing, feature analysis, and comparison of two classical classification algorithms:

- Logistic Regression
- Linear Discriminant Analysis (LDA)

---

##  Project Overview

Diabetes is a common chronic disease where early identification of high-risk individuals can support timely medical intervention.

In this project, machine learning classification techniques are applied to predict the diabetes outcome (`0` = non-diabetic, `1` = diabetic) using patient-level diagnostic features.

The project follows a complete machine learning workflow:

**Data Loading → Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Preprocessing → Model Training → Model Evaluation → Model Comparison**

---

##  Problem Statement

Build a binary classification model that predicts whether a patient has diabetes based on the available medical and demographic attributes.

### Target Variable

| Outcome | Meaning |
|---|---|
| `0` | Non-Diabetic |
| `1` | Diabetic |

---

## Dataset

The dataset contains **768 observations** and **9 columns**.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age of the patient |
| Outcome | Diabetes classification target |

The dataset contains no explicit null values initially. However, several features contain `0` values that are treated as invalid/missing values for analysis.

---

## Exploratory Data Analysis

The exploratory analysis included:

- Dataset structure and data types
- Descriptive statistics
- Missing-value inspection
- Invalid zero-value analysis
- Target variable distribution
- Feature distributions
- Boxplot-based outlier analysis
- Outcome-wise feature comparison
- Correlation analysis

### Invalid Zero Values

Zero values were identified in the following features:

| Feature | Zero Values |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

These values were treated as missing values rather than valid measurements.

---

## Data Preprocessing

### 1. Handling Invalid Values

The following features were checked for invalid zero values:

```python
invalid_zero_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

The zero values were replaced with NaN.

2. Missing Value Imputation

After identifying the invalid values, missing values were replaced using the median of each respective feature.

Median imputation was selected to provide a robust approach in the presence of skewed distributions and outliers.

3. Train-Test Split

The dataset was divided into training and testing sets before model development.

4. Feature Scaling

StandardScaler was used to standardize the input features before training Logistic Regression and LDA models.

Machine Learning Models

Two classification algorithms were trained and evaluated.

1. Logistic Regression

Logistic Regression was used as the primary baseline classification model.

It predicts the probability of the target belonging to the diabetic class and converts the probability into a binary classification.

2. Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis was used as a second classification approach for comparison with Logistic Regression.

📈 Model Performance

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve
AUC
Model Comparison
Model	Training Accuracy	Test Accuracy	AUC
Logistic Regression	79.64%	70.78%	0.813
LDA	79.15%	70.13%	0.813
Classification Performance
Logistic Regression
Class	Precision	Recall	F1-Score
Non-Diabetic (0)	0.75	0.82	0.78
Diabetic (1)	0.60	0.50	0.55

Overall Test Accuracy: 70.78%

LDA
Class	Precision	Recall	F1-Score
Non-Diabetic (0)	0.75	0.82	0.78
Diabetic (1)	0.59	0.48	0.53

Overall Test Accuracy: 70.13%

ROC Curve & AUC

ROC curves were generated to compare the ability of both models to distinguish between diabetic and non-diabetic patients across different classification thresholds.

The resulting AUC scores were:

Logistic Regression: 0.813
LDA: 0.813

The two models therefore demonstrated very similar discrimination performance based on AUC, with Logistic Regression having a very small numerical advantage.

Key Findings
1. Glucose is an important predictor

The exploratory analysis showed noticeably higher glucose levels among diabetic patients compared with non-diabetic patients.

Correlation analysis also identified Glucose as having the strongest linear relationship with the diabetes outcome among the analyzed features.

2. Invalid zero values require treatment

Several medical measurements contained zero values that were treated as missing observations.

Median imputation was subsequently used to handle these missing values.

3. Logistic Regression performed slightly better

Logistic Regression achieved:

Higher test accuracy: 70.78% vs 70.13%
Slightly higher AUC numerically: 0.81296 vs 0.81259
Better recall and F1-score for the diabetic class

However, the difference between the two models is small.

4. Class 1 prediction remains challenging

Both models performed better at identifying non-diabetic patients than diabetic patients.

For the diabetic class, recall was:

Logistic Regression: 0.50
LDA: 0.48

This indicates that there is room for improvement in identifying diabetic cases.

Feature Importance

Feature coefficients from the Logistic Regression model were analyzed to understand the relative contribution of the input features to the prediction.

This provides an additional interpretation layer beyond overall model performance.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Project Structure
Pima_Indians_Diabetes_prediction/
│
├── pima_diabetes_analysis.ipynb
├── diabetes.csv
└── README.md
How to Run the Project
1. Clone the repository
git clone <your-repository-url>
2. Navigate to the project directory
cd Pima_Indians_Diabetes_prediction
3. Install the required libraries
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
4. Launch Jupyter Notebook
jupyter notebook

Open the notebook and run the cells sequentially.

Limitations

This project is intended as a machine learning learning and portfolio project and should not be considered a clinical diagnostic system.

The current models also show relatively lower recall for the diabetic class, meaning some diabetic cases may be classified as non-diabetic.

Further improvements could include:

Hyperparameter tuning
Cross-validation
Additional classification algorithms
Threshold optimization
More detailed feature engineering
Ensemble learning
More extensive model evaluation
Future Improvements

Potential next steps include experimenting with:

Decision Trees
Random Forest
Support Vector Machines
K-Nearest Neighbors
Gradient Boosting
XGBoost
Ensemble methods

Model selection could also be optimized based on the project's primary objective, particularly if improving diabetic-case recall is more important than maximizing overall accuracy.

Project Goal

The primary goal of this project is to demonstrate a complete end-to-end machine learning classification workflow, from data exploration and preprocessing to model training, evaluation, interpretation, and comparison.
