import pandas as pd

# Load the dataset
df = pd.read_csv('../CSVfiles/balanced_mbti.csv')

# Total number of records
total_records = len(df)

# Function to count words in the 'posts' column
def count_words(text):
    if pd.isna(text): # ensure text is not NaN
        return 0
    return len(str(text).split())

# Add a new column with the number of words in each record
df['word_count'] = df['posts'].apply(count_words)

# Calculate the average number of words per record
average_words = df['word_count'].mean()

# Calculate the median number of words per record
median_words = df['word_count'].median()

# Print the results
print("\nTotal number of records:", total_records)
print(f"Average number of words per record: {average_words:.2f}")
print(f"Median number of words per record: {median_words}")