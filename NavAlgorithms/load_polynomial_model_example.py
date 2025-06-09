import pickle
import pandas as pd

# Load the saved polynomial regression model
with open("polynomial_model.pkl", "rb") as f:
    poly_model = pickle.load(f)

# Example: Prepare input data (replace with your own data)
# Here, we use a DataFrame with columns 'vx_input' and 'vy_input'
example_data = pd.DataFrame({
    'vx_input': [450, 412, 500],
    'vy_input': [0, 0, 33]
})

# Predict using the loaded model
predictions = poly_model.predict(example_data)
print("Predictions (vx_output, vy_output):")
print(predictions)
