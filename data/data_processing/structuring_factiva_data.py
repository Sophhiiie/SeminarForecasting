import os
import pandas as pd
from striprtf.striprtf import rtf_to_text
from concurrent.futures import ThreadPoolExecutor
import re

# ----------------------------
# STEP 1: Setup and load files
# ----------------------------
factiva1 = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/Factiva-data-short-version/Factiva-20250324-1505"
factiva2 = "/Users/sophiemerks/Desktop/Seminar/Code/Seminar code/data/Factiva-data-short-version/Factiva-20250325-1240_(2)"

def get_rtf_files(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.rtf')]

all_files = get_rtf_files(factiva1) + get_rtf_files(factiva2)

# ----------------------------
# STEP 2: Read and extract RTF content
# ----------------------------
def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            text = rtf_to_text(content)
            return {'filename': os.path.basename(filepath), 'text': text}
    except Exception as e:
        return {'filename': os.path.basename(filepath), 'text': '', 'error': str(e)}

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_file, all_files))

df = pd.DataFrame(results)

# ----------------------------
# STEP 3: Extract structured articles from text
# ----------------------------

# Regex pattern for article structure
article_pattern = re.compile(
    r'(?P<section_and_title>.+?)\nThe Wall Street Journal,\s+(?P<date>\d{1,2} \w+ \d{4}),\s+'
    r'(?P<words>\d+)\s+words,\s+By (?P<author>.+?),\s+\(English\)\n',
    re.DOTALL
)

def extract_articles_from_text(filename, text):
    articles = []
    matches = list(article_pattern.finditer(text))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        section_and_title = match.group('section_and_title').strip()

        # Optional: split section and title if possible
        if ' --- ' in section_and_title:
            section, title = section_and_title.split(' --- ', 1)
        else:
            section = None
            title = section_and_title

        articles.append({
            'filename': filename,
            'section': section,
            'title': title.strip(),
            'date': match.group('date').strip(),
            'word_count': int(match.group('words')),
            'author': match.group('author').strip(),
            'body': body
        })

    return articles

# ----------------------------
# STEP 4: Apply to all files and export
# ----------------------------
all_articles = []
for _, row in df.iterrows():
    all_articles.extend(extract_articles_from_text(row['filename'], row['text']))

articles_df = pd.DataFrame(all_articles)
articles_df['date'] = pd.to_datetime(articles_df['date'], format='%d %B %Y', errors='coerce')

# Save to CSV
articles_df.to_csv('factiva_articles_split.csv', index=False)

print("✅ Done! Extracted", len(articles_df), "articles.")
