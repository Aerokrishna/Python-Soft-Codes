import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load CSV
df = pd.read_csv("combined_dataset.csv")

# Features and targets
X = df[['vx_input', 'vy_input']]
y = df[['vx_output', 'vy_output']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial regression pipeline (try degree 2, 3, etc.)
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(X_train, y_train)

# Save the trained model
with open("polynomial_model.pkl", "wb") as f:
    pickle.dump(poly_model, f)

# Predict
y_pred = poly_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Polynomial Degree: {degree}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# vx plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test['vx_output'], y_pred[:, 0], alpha=0.6, label='vx')
plt.plot([y_test['vx_output'].min(), y_test['vx_output'].max()],
         [y_test['vx_output'].min(), y_test['vx_output'].max()], 'k--')
plt.xlabel("Actual vx_output")
plt.ylabel("Predicted vx_output")
plt.title("vx: Actual vs Predicted (Polynomial)")
plt.grid(True)

# vy plot
plt.subplot(1, 2, 2)
plt.scatter(y_test['vy_output'], y_pred[:, 1], alpha=0.6, label='vy', color='orange')
plt.plot([y_test['vy_output'].min(), y_test['vy_output'].max()],
         [y_test['vy_output'].min(), y_test['vy_output'].max()], 'k--')
plt.xlabel("Actual vy_output")
plt.ylabel("Predicted vy_output")
plt.title("vy: Actual vs Predicted (Polynomial)")
plt.grid(True)

plt.tight_layout()
plt.show()
