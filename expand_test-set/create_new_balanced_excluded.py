"""
This script moves 5 random ENTJ and 5 random ESTJ entries
in order to expand the test set to 30 records. (was 20 records before)
"""
import pandas as pd

# load the existing files
balanced_df = pd.read_csv('../CSVfiles/balanced_mbti.csv')
excluded_df = pd.read_csv('../CSVfiles/excluded_mbti.csv')

# randomly select 5 ENTJ and 5 ESTJ rows from balanced_df to move
entj_to_move = balanced_df[balanced_df['type'] == 'ENTJ'].sample(n=5, random_state=1)
estj_to_move = balanced_df[balanced_df['type'] == 'ESTJ'].sample(n=5, random_state=1)

# combine the selected rows
rows_to_move = pd.concat([entj_to_move, estj_to_move])

# remove them from balanced_df
balanced_df = balanced_df.drop(rows_to_move.index)

# add them to excluded_df
excluded_df = pd.concat([excluded_df, rows_to_move])

# save the updated files
balanced_df.to_csv('new_balanced_mbti.csv', index=False)
excluded_df.to_csv('new_excluded_mbti.csv', index=False)

print("files updated: 5 ENTJ and 5 ESTJ moved from balanced to excluded.")