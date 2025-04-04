# This file transforms the raw data into an organized pandas table


############# Processing the Factiva-data-short-version




############# Processing the etfs_per_sector_2021
# Define the folder containing ETF CSV files
etf_folder = "data/etfs_per_sector_2021"

# List all .csv files in the folder
etf_files = [f for f in os.listdir(etf_folder) if f.endswith(".csv")]

# Create a list to hold individual DataFrames
all_etfs = []

# Read and store each ETF CSV
for filename in etf_files:
    filepath = os.path.join(etf_folder, filename)

    # Read the current CSV file
    df = pd.read_csv(filepath)

    # Add a column to indicate the ETF name (based on the filename)
    df['ETF'] = filename.replace(" ETF Stock Price History.csv", "")

    # Rename columns for clarity (if needed)
    df.columns = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume', 'Change %', 'ETF']

    # Convert 'Date' column to datetime for consistency
    df['Date'] = pd.to_datetime(df['Date'])

    # Append to the list
    all_etfs.append(df)

# Combine all ETF DataFrames into one large DataFrame
combined_df = pd.concat(all_etfs, ignore_index=True)

# Save the combined DataFrame to a CSV file
output_path = "data/processed_all_etfs.csv"
combined_df.to_csv(output_path, index=False)



############ Processing the Reddit data

