import pandas as pd

# Input and output file paths
input_file_path = '/home/rravindra0463@id.sdsu.edu/models/datasets/trivia_qa_web.csv'  # Replace with the path to your dataset
output_file_path = 'subset_trivia_qa_web.csv'  # Replace with the desired path for the subset

# Number of rows to extract
num_rows_to_extract = 50
# Columns to keep
columns_to_keep = ["question", "answer"]

# Read the first 50 rows from the dataset
df = pd.read_csv(input_file_path, nrows=num_rows_to_extract)

# Keep only specified columns
df = df[columns_to_keep]

# Save the subset to a new CSV file
df.to_csv(output_file_path, index=False)

print(f"Subset containing the first {num_rows_to_extract} rows saved to {output_file_path}.")
