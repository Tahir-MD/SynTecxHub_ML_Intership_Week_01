# SynTecxHub_ML_Intership_Week_01
# House Price Prediction - Machine Learning Project

A complete machine learning solution for predicting house prices using Linear Regression. This project implements a full data science pipeline from data loading and preprocessing to model training, evaluation, and deployment.

# 🏠 Project Overview

This project predicts house prices based on various features such as location, number of rooms, population density, and median income. It demonstrates end-to-end machine learning workflow including data exploration, feature engineering, model training, and evaluation.

# 📊 Dataset

The project uses the California Housing Dataset which contains:

· 20,640 samples of housing districts
· 8 numerical features + 1 categorical feature
· Target variable: median_house_value

# Features:

· longitude, latitude: Geographic coordinates
· housing_median_age: Median age of houses
· total_rooms: Total number of rooms
· total_bedrooms: Total number of bedrooms
· population: Total population in block
· households: Total number of households
· median_income: Median income of households
· ocean_proximity: Proximity to ocean (categorical)

# 🚀 Features

· Complete Data Pipeline: Loading, cleaning, and preprocessing
· Exploratory Data Analysis: Visualizations and statistical analysis
· Feature Engineering: Handling categorical variables, correlation analysis
· Model Training: Linear Regression implementation
· Model Evaluation: RMSE and R² score calculation
· Results Visualization: Actual vs Predicted plots, residual analysis
· Model Persistence: Save trained model using pickle

# 📈 Results

The trained Linear Regression model achieves:

· RMSE (Root Mean Square Error): ~$68,000
· R² Score: ~0.65
· Interpretable coefficients showing feature importance

# 🛠️ Installation & Requirements

Prerequisites

· Python 3.8+
· pip package manager

Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
