import os
import pandas as pd
from striprtf.striprtf import rtf_to_text
from concurrent.futures import ThreadPoolExecutor
import re

# Define folders
factiva1 = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/Factiva-data-short-version/Factiva-20250324-1505"
factiva2 = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/Factiva-data-short-version/Factiva-20250325-1240_(2)"

# Get list of .rtf files from both folders
def get_rtf_files(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.rtf')]

all_files = get_rtf_files(factiva1) + get_rtf_files(factiva2)

# Function to read and convert RTF content
def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            text = rtf_to_text(content)
            return {'filename': os.path.basename(filepath), 'text': text}
    except Exception as e:
        return {'filename': os.path.basename(filepath), 'text': '', 'error': str(e)}

# Use multithreading for faster processing
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_file, all_files))

# Create DataFrame
df = pd.DataFrame(results)

# Regex pattern for dates like '30 June 2021' or '5 July 2020'
date_pattern = re.compile(r'\b\d{1,2} [A-Z][a-z]+ \d{4}\b')

def extract_date(text):
    match = date_pattern.search(text)
    return match.group(0) if match else None

# Add date column
df['date'] = df['text'].apply(extract_date)

# Apply the function to add a 'date' column
df['date'] = df['text'].apply(extract_date)

# Optional: convert string date to datetime object
df['date'] = pd.to_datetime(df['date'], format='%d %B %Y', errors='coerce')

# Optional: save to CSV
df.to_csv('factiva_combined2.csv', index=False)