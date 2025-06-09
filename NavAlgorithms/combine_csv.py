import pandas as pd

# List of your CSV file paths
csv_files = ["data_collection_output.csv", "data_collection_output_2.csv", "data_collection_output_3.csv", "data_collection_output_4.csv", "data_collection_output_5.csv"]

# Read and concatenate all
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Save to new CSV if needed
combined_df.to_csv("combined_dataset.csv", index=False)
