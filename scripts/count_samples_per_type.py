import pandas as pd

# load the data
CVS_file = '../try_things/PersonalityCafe_mbti.csv'
df = pd.read_csv(CVS_file)

# Count the number of records for each personality type.
personality_counts = df['type'].value_counts()

# print the result
print(personality_counts)
