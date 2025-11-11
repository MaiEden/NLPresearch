import pandas as pd
from dataProccesing.Models.LR_model import LRModel
from dataProccesing.Models.SVM_model import SVMModel

# read the data
mbti_data = pd.read_csv("../CSVfiles/balanced_mbti.csv")

def remove_types(text):
    # remove types names from the data
    types = ["istj", "isfj", "infj", "intj", "istp", "isfp", "infp", "intp",
             "estp", "esfp", "enfp", "entp", "estj", "esfj", "enfj", "entj"]
    words = text.split()
    words = [w for w in words if not w in types]
    return ' '.join(words)

# mbti_data['posts'] = mbti_data['posts'].apply(remove_types) # clean the data set
# # save it in order to save run-time
# mbti_data.to_csv("../CSVfiles/mbti_no_types_names.csv",index=False)

# # run LR on the cleaned data
# LR_model = LRModel("../CSVfiles/mbti_no_types_names.csv", cleaned=True)
#
# LR_model.prediction()

# run SVM on the cleaned data
SVM_model = SVMModel("../CSVfiles/balanced_mbti.csv", cleaned=True)

SVM_model.prediction()