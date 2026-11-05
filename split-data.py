import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. Read the raw header string from the original dataset
with open('dataset.csv', 'r', encoding='utf-8') as f:
    raw_header = f.readline().strip()

# Clean the header: keep only alphanumeric characters, spaces, and semicolons
cleaned_header = re.sub(r'[^a-zA-Z0-9\s;]', '', raw_header)
column_names = cleaned_header.split(';')

# 2. Read all remaining lines (the data rows) from the original file
with open('dataset.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:] # Skip the original header line

# 3. Write back the cleaned header and the data lines into dataset.csv
with open('dataset.csv', 'w', encoding='utf-8') as f:
    f.write(cleaned_header + '\n')
    f.writelines(lines)

print("Updated and cleaned header successfully written back to dataset.csv!")

# 4. Load the now-clean dataset and split it into train and test sets
df = pd.read_csv('dataset.csv', sep=';', encoding='utf-8')

# Clean up any trailing/leading whitespaces from column headers just in case
df.columns = df.columns.str.strip()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. Save the split datasets
train_df.to_csv('train_data.csv', sep=';', index=False)
test_df.to_csv('test_data.csv', sep=';', index=False)
print("Writing train_data.csv and test_data.csv with 80:20 split")
print(f"Training set rows: {len(train_df)}, Testing set rows: {len(test_df)}")
