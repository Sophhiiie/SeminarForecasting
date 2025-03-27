import os
import pandas as pd
import re
from striprtf.striprtf import rtf_to_text

# This is file is created to organize the raw data obtained from different sources


# Set the base path to your Factiva folders
base_folder = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/Factiva-data-short-version"


# Create an empty list to store article data
articles = []

# Loop through both folders
for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".rtf"):
                file_path = os.path.join(subfolder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    try:
                        rtf_content = file.read()
                        text = rtf_to_text(rtf_content)

                        lines = text.strip().split("\n")
                        title = lines[0].strip() if lines else "No Title"

                        # Try to extract a date in format like "22 July 2021"
                        date_match = re.search(r"\d{1,2} \w+ \d{4}", text)
                        date = date_match.group() if date_match else "Unknown"

                        article_text = "\n".join(lines[1:]).strip()

                        articles.append({
                            "file": filename,
                            "title": title,
                            "date": date,
                            "text": article_text
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame(articles)

# Preview
print(df.head())

# Optional: Save to CSV
df.to_csv("data/processed_factiva_articles.csv", index=False)

