# Gold Price Prediction

## Project Overview

This project aims to predict gold prices using various machine learning models. The project workflow includes data preprocessing, feature analysis, model development, and hyperparameter tuning.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Analysis](#feature-analysis)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

Gold price prediction is crucial for financial markets and investors. This project leverages machine learning techniques to predict gold prices based on various economic indicators.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Gradient boosting regression
- **GridSearchCV**: Hyperparameter tuning

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Cleaning**:
   - Checked for and handled missing values.

3. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Feature Analysis

1. **Correlation Analysis**:
   - Visualized and identified significant correlations between features.

2. **Visualization**:
   - Used scatter plots and density plots to understand feature distributions and relationships.

## Modeling

1. **Linear Regression**:
   - Built a simple linear regression model and evaluated its performance.

2. **Random Forest Regressor**:
   - Built a Random Forest model and optimized it using GridSearchCV.

3. **AdaBoost Regressor**:
   - Built an AdaBoost model with a Decision Tree base estimator.

4. **XGBoost**:
   - Implemented an XGBoost model for enhanced prediction accuracy.

## Results

- **Linear Regression**:
  - Training Accuracy: 0.76
  - Test Accuracy: 0.75

- **Random Forest Regressor**:
  - Training Accuracy: 0.99
  - Test Accuracy: 0.98

- **AdaBoost Regressor**:
  - Training Accuracy: 0.99
  - Test Accuracy: 0.98

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/gold-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd gold-price-prediction
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict gold prices based on new data.

## Conclusion

This project demonstrates the use of various machine learning models to predict gold prices. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into the factors influencing gold prices.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Load data
data = pd.read_csv("D:\\Courses language programming\\Machine Learning\\Folder Machine Learning\\Gold_priceData\\gld_price_data.csv")

# Data information
data.info()
data.describe()
data.isnull().sum()

# Data visualization
plt.figure(figsize=(5, 5))
sns.heatmap(data=data.corr(), annot=True, fmt="0.2f", square=True, cbar=True, cmap="Blues")

plt.figure(figsize=(10, 5))
z = plt.scatter(x=data["EUR/USD"], y=data["SLV"], c=data["USO"], cmap=plt.get_cmap("jet"), marker="o")
plt.colorbar(z)
plt.title("The Relation Between (Eur/USD, SLV, USO)")
plt.xlabel("EUR/USD")
plt.ylabel("SLV & USO")
plt.grid()

plt.figure(figsize=(10, 5))
z = plt.scatter(x=data["EUR/USD"], y=data["GLD"], c=data["SPX"], cmap=plt.get_cmap("jet"), marker="o")
plt.colorbar(z)
plt.title("The Relation Between (Eur/USD, SPX, GLD)")
plt.xlabel("EUR/USD")
plt.ylabel("SPX & GLD")
plt.grid()

data.boxplot(figsize=(20, 15))
data["SPX"].plot(kind="density", figsize=(5, 2.5))
data["GLD"].plot(kind="density", figsize=(5, 2.5))
data["USO"].plot(kind="density", figsize=(5, 2.5))

# Split data
X = data.drop(columns=["EUR/USD", "Date"], axis=1)
Y = data["EUR/USD"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Linear Regression
Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)
print("Linear Regression Train Score:", Lin_reg.score(x_train, y_train))
print("Linear Regression Test Score:", Lin_reg.score(x_test, y_test))

# Random Forest Regressor
model_RF = RandomForestRegressor(n_estimators=2000, max_depth=500)
model_RF.fit(x_train, y_train)
print("Random Forest Train Score:", model_RF.score(x_train, y_train))
print("Random Forest Test Score:", model_RF.score(x_test, y_test))

param = {"n_estimators": np.arange(10, 25, 2),
         "max_depth": np.arange(9, 11, 1),
         "min_samples_split": [2, 3, 4]}
new_model = GridSearchCV(estimator=model_RF, param_grid=param, cv=10, n_jobs=-1)
new_model.fit(x_train, y_train)
print("Best Estimator:", new_model.best_estimator_)

# AdaBoost Regressor
model_AD1 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=500),
                              n_estimators=1000,
                              learning_rate=0.5,
                              loss="exponential")
model_AD1.fit(x_train, y_train)
print("AdaBoost Train Score:", model_AD1.score(x_train, y_train))
print("AdaBoost Test Score:", model_AD1.score(x_test, y_test))

# XGBoost Regressor
model_xgb = xgb.XGBRFRegressor(n_estimators=500, max_depth=50, max_delta_step=2, learning_rate=1)
model_xgb.fit(x_train, y_train)
print("XGBoost Train Score:", model_xgb.score(x_train, y_train))
print("XGBoost Test Score:", model_xgb.score(x_test, y_test))
```
