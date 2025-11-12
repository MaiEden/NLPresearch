"""
This script runs the Logistic Regression model for MBTI personality type prediction.
data-set: balanced_mbti.csv
for additional testing: excluded_mbti.csv (20 records)
"""
from Models.LR_model import LRModel

LR_model = LRModel("../CSVfiles/balanced_mbti.csv", cleaned=True)

LR_model.prediction()

LR_model.predict_additional_posts("../CSVfiles/excluded_mbti.csv", num_records=20)


# Get and print the top positive and negative feature weights
top_pos, top_neg = LR_model.get_feature_weights(top_n=30)

print("Assertive words:")
print(top_pos)

print("\nNon-assertive words:")
print(top_neg)

print(LR_model.get_word_weight("leader"))
print(LR_model.get_word_weight("maybe"))