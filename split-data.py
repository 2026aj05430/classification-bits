import pandas as pd

# Load data
df = pd.read_csv('your_dataset.csv')

# Shuffle the dataset randomly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the split point (e.g., 80% for training)
split_idx = int(0.8 * len(df_shuffled))

# Divide into train and test
train_df = df_shuffled.iloc[:split_idx]
test_df = df_shuffled.iloc[split_idx:]

# Save to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
