import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("combined_dataset.csv")

# Sample a few points
sample = df.sample(500)

# Scatter plot: vx_input vs vx_output
plt.subplot(1, 2, 1)
plt.scatter(sample['vx_input'], sample['vx_output'], alpha=0.6)
plt.xlabel("vx_input")
plt.ylabel("vx_output")
plt.title("vx_input vs vx_output")
plt.grid(True)

# Scatter plot: vy_input vs vy_output
plt.subplot(1, 2, 2)
plt.scatter(sample['vy_input'], sample['vy_output'], alpha=0.6, color='orange')
plt.xlabel("vy_input")
plt.ylabel("vy_output")
plt.title("vy_input vs vy_output")
plt.grid(True)

plt.tight_layout()
plt.show()
