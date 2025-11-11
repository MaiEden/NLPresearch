"""
This script runs the Logistic Regression model for MBTI personality type prediction.
data-set: balanced_mbti.csv
for additional testing: excluded_mbti.csv (20 records)
"""
from dataProccesing.Models.LR_model import LRModel

LR_model = LRModel("../CSVfiles/balanced_mbti.csv", cleaned=True)

LR_model.prediction()

LR_model.predict_additional_posts("../CSVfiles/excluded_mbti.csv", num_records=20)