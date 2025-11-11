"""
This script runs the SVM for MBTI personality type prediction.
data-set: balanced_mbti.csv
for additional testing: excluded_mbti.csv (20 records)
"""
from dataProccesing.Models.SVM_model import SVMModel

SVM_model = SVMModel("../CSVfiles/balanced_mbti.csv", cleaned=True)

SVM_model.prediction()

SVM_model.predict_additional_posts("../CSVfiles/excluded_mbti.csv", num_records=20)