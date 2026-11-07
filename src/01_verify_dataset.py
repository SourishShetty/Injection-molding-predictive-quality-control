
import pandas as pd

# Step 1: Define the dataset location
file_path = "data/Dataset.csv"

# Step 2: Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Step 3: Display basic dataset information
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Step 4: Display the first five rows
print("\nFirst five rows:")
print(df.head())

# Step 5: Display all column names
print("\nColumn names:")
print(df.columns.tolist())
