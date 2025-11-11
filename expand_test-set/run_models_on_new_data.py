from dataProccesing.Models.SVM_model import SVMModel
from dataProccesing.Models.LR_model import LRModel

def run_SVM():
    SVM_model = SVMModel("new_balanced_mbti.csv", cleaned=True)
    SVM_model.prediction()
    SVM_model.predict_additional_posts("new_excluded_mbti.csv", num_records=30)

def run_LR():
    LR_model = LRModel("new_balanced_mbti.csv", cleaned=True)
    LR_model.prediction()
    LR_model.predict_additional_posts("new_excluded_mbti.csv", num_records=30)

#Run one of the models on the new data-set
model = input("Enter the model to run (SVM or LR): ").upper()
if model == "SVM":
    run_SVM()
elif model == "LR":
    run_LR()
else:
    print("Invalid model choice. Please enter 'SVM' or 'LR'.")