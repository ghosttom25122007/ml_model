"""
Student Score Predictor - Model Training Script
--------------------------------------------------
This script:
1. Creates a small dataset of study hours vs exam scores
2. Trains a Linear Regression model to find the relationship
3. Evaluates the model
4. Saves the trained model to disk for deployment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Create the dataset (hours studied vs score obtained)
# In a real project you might load this from a CSV, but this is a
# realistic, clean sample dataset good for a demo.
data = {
    "hours_studied": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
                       6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 2.2,
                       3.8, 5.2, 6.8, 7.2, 4.4],
    "score": [10, 16, 22, 28, 35, 40, 47, 53, 58, 63,
              70, 74, 80, 84, 89, 92, 95, 97, 99, 25,
              43, 60, 76, 82, 50]
}

df = pd.DataFrame(data)

X = df[["hours_studied"]]   # feature (2D, sklearn requires this shape)
y = df["score"]             # target

# 2. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully!")
print(f"R² Score: {r2:.3f}  (closer to 1.0 = better fit)")
print(f"Mean Absolute Error: {mae:.2f} marks")
print(f"Learned equation: score = {model.coef_[0]:.2f} * hours + {model.intercept_:.2f}")

# 5. Save the trained model
joblib.dump(model, "score_model.joblib")
print("Model saved as score_model.joblib")
