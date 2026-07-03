#!/usr/bin/env python
# coding: utf-8

# In[7]:


# ==========================================================
# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import joblib

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("AAA.csv")

print("Original Shape :", df.shape)

# ==========================================================
# REMOVE EMPTY COLUMNS
# ==========================================================

df = df.dropna(axis=1, how='all')

# ==========================================================
# REMOVE EMPTY ROWS
# ==========================================================

df = df.dropna(how='all')

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

df = df.drop_duplicates()

# ==========================================================
# RESET INDEX
# ==========================================================

df.reset_index(drop=True, inplace=True)

print("Shape after cleaning :", df.shape)

# ==========================================================
# CHECK MISSING VALUES
# ==========================================================

print(df.isnull().sum())

# ==========================================================
# HANDLE REMAINING MISSING VALUES
# ==========================================================

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# ==========================================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================================

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nProcessed Dataset\n")
print(df.head())

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop("price", axis=1)
y = df["price"]

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# MODEL EVALUATION FUNCTION
# ==========================================================

def evaluate(model, name):

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    r2 = r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))

    print("\n====================================")
    print(name)
    print("====================================")
    print("R2 Score :", round(r2,4))
    print("MAE      :", round(mae,2))
    print("RMSE     :", round(rmse,2))

    return prediction

# ==========================================================
# LINEAR REGRESSION
# ==========================================================

linear = LinearRegression()
pred_lr = evaluate(linear, "Linear Regression")

# ==========================================================
# DECISION TREE
# ==========================================================

tree = DecisionTreeRegressor(random_state=42)
pred_tree = evaluate(tree, "Decision Tree")

# ==========================================================
# RANDOM FOREST
# ==========================================================

forest = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

pred_rf = evaluate(forest, "Random Forest")

# ==========================================================
# GRADIENT BOOSTING
# ==========================================================

gbr = GradientBoostingRegressor(random_state=42)
pred_gbr = evaluate(gbr, "Gradient Boosting")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# ==========================================================
# PLOT FEATURE IMPORTANCE
# ==========================================================

plt.figure(figsize=(8,6))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

plt.figure(figsize=(7,7))
plt.scatter(y_test, pred_rf)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest Prediction")
plt.grid(True)
plt.show()

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(forest, "HousePriceModel.pkl")
joblib.dump(scaler, "Scaler.pkl")

print("\nModel Saved Successfully.")

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("HousePriceModel.pkl")
scaler = joblib.load("Scaler.pkl")

# ==========================================================
# PREDICT NEW HOUSE
# ==========================================================

new_house = pd.DataFrame({

    "area":[8000],
    "bedrooms":[4],
    "bathrooms":[3],
    "stories":[2],
    "mainroad":[1],
    "guestroom":[1],
    "basement":[1],
    "hotwaterheating":[0],
    "airconditioning":[1],
    "parking":[2],
    "prefarea":[1],
    "furnishingstatus":[2]

})

new_house_scaled = scaler.transform(new_house)

prediction = model.predict(new_house_scaled)

print("\n====================================")
print("Predicted House Price")
print("====================================")
print("Estimated Price :", round(prediction[0],2))


# In[ ]:




