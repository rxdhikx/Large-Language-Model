import pandas as pd
df = pd.read_parquet('dataset_file_name.parquet')
df.to_csv('dataset_file_name.csv')

df.head()
