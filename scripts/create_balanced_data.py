import pandas as pd

# load the data
csv_file = '../CSVfiles/MBTI 500.csv'
df = pd.read_csv(csv_file)

# take assertive types (ENTJ and ESTJ)
entj_df = df[df['type'] == 'ENTJ']
estj_df = df[df['type'] == 'ESTJ']

# keep 20 records outside the balanced data (10 from each type)
entj_reserved = entj_df.sample(n=10, random_state=42)
estj_reserved = estj_df.sample(n=10, random_state=42)

# remove the reserved records
entj_df = entj_df.drop(entj_reserved.index)
estj_df = estj_df.drop(estj_reserved.index)

# calc the total number of ENTJ and ESTJ rows
total_entj_estj = len(entj_df) + len(estj_df)

# take all the non-assertive types
other_df = df[(df['type'] != 'ENTJ') & (df['type'] != 'ESTJ')]

# counts the number of records for each other personality type
other_counts = other_df['type'].value_counts()

# splitting the total number of ENTJ and ESTJ rows equally among
# other types (for balance)
samples_per_other_type = total_entj_estj // len(other_counts)

# create a new empty DataFrame
selected_other_df = pd.DataFrame()

# take the records from each non-assertive type
for personality_type in other_counts.index:
    type_df = other_df[other_df['type'] == personality_type]
    num_samples_for_type = min(samples_per_other_type, len(type_df))
    selected_other_df = pd.concat([selected_other_df, type_df.sample(n=num_samples_for_type, random_state=42)])

# unify all selected data
final_df = pd.concat([entj_df, estj_df, selected_other_df])

# save the data
final_df.to_csv('../CSVfiles/balanced_mbti.csv', index=False)

# get the excluded data
excluded_df = df.drop(final_df.index)
# save all excluded data
excluded_df.to_csv('../CSVfiles/excluded_mbti.csv', index=False)

print("file 'balanced_mbti.csv' created successfully.")
print("file 'excluded_mbti.csv' created successfully.")