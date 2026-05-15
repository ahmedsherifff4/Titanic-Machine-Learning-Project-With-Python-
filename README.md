# 🚢 Titanic Survival Prediction Project

A complete Machine Learning project for predicting passenger survival on the Titanic dataset using multiple classification algorithms, feature engineering, hyperparameter tuning, data visualization, and a deployed interactive Streamlit application.

---

# 📌 Project Overview

This project aims to predict whether a passenger survived the Titanic disaster based on passenger information such as:

- Age
- Gender
- Passenger Class
- Fare
- Family Size
- Embarkation Port
- Passenger Title

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training & Evaluation
- Hyperparameter Tuning
- Model Comparison
- Streamlit Deployment

---

# 📂 Dataset

The project uses the classic Titanic dataset:

- `Titanic_train.csv`
- `Titanic_test.csv`

Main features include:

| Feature | Description |
|---|---|
| Pclass | Passenger class |
| Sex | Gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation |
| Survived | Target variable |

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Filling missing `Age` and `Fare` using median values
- Filling missing `Embarked` values using mode
- Removing unnecessary columns
- Encoding categorical features
- Standard Scaling
- Feature Engineering

---

# ⚙️ Feature Engineering

Additional features were created to improve model performance.

## 🔹 Extracted Passenger Titles

Titles were extracted from passenger names:

- Mr
- Mrs
- Miss
- Master
- Rare

---

## 🔹 Family Features

Created:

- `FamilySize`
- `IsAlone`

---

## 🔹 Age Binning

Passengers were grouped into age categories:

- Child
- Teen
- Young
- Adult
- Senior

---

# 📊 Exploratory Data Analysis (EDA)

Several visualizations were used to better understand the dataset:

- Survival Distribution
- Survival by Gender
- Survival by Passenger Class
- Survival by Embarkation Port
- Age Distribution
- Fare Distribution
- Correlation Heatmaps
- Confusion Matrices
- Learning Curves

---

# 🤖 Machine Learning Models

The following classification models were trained and evaluated:

| Model | Used |
|---|---|
| Logistic Regression | ✅ |
| Support Vector Machine (SVM) | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| K-Nearest Neighbors (KNN) | ✅ |

---

# 🔧 Hyperparameter Tuning

GridSearchCV and manual tuning were applied to improve performance for:

- Logistic Regression
- SVM
- Decision Tree

---

# 📈 Model Evaluation

Models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Learning Curves
- Overfitting Gap Analysis

---

# 🏆 Best Model Selection

The project automatically selects the best-performing model based on validation accuracy.

```python
best_name  = max(val_accs, key=val_accs.get)
best_model = models_dict[best_name]
