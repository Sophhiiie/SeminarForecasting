import os
import pandas as pd

# Processing the sector specific ETFs


etf_folder = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/etfs_per_sector_2021"

# List all .csv files in the folder
etf_files = [f for f in os.listdir(etf_folder) if f.endswith(".csv")]


etf_files = [f for f in os.listdir(etf_folder) if f.endswith(".csv")]

# Create a list to hold individual DataFrames
all_etfs = []

# Read and store each ETF CSV
for filename in etf_files:
    filepath = os.path.join(etf_folder, filename)

    # Read the current CSV file
    df = pd.read_csv(filepath)

    # Extract the ETF name from the filename (e.g., XLB from XLB ETF Stock Price History.csv)
    etf_name = filename.split(' ')[0]

    # Add a column to indicate the ETF name
    df['ETF'] = etf_name

    # Rename columns for clarity (if needed)
    df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume', 'Change %', 'ETF']

    # Convert 'Date' column to datetime for consistency
    df['Date'] = pd.to_datetime(df['Date'])

    # Append to the list
    all_etfs.append(df)

# Combine all ETF DataFrames into one large DataFrame
combined_df = pd.concat(all_etfs, ignore_index=True)

# Check the first few rows of the combined data
print(combined_df.head())

# Save the combined DataFrame to a CSV file
output_path = "data/processed_all_etfs.csv"
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined ETF data saved to {output_path}")