import pandas as pd
from nltk.corpus import stopwords

from Models.LR_model import LRModel
from Models.SVM_model import SVMModel

# read the data
mbti_data = pd.read_csv("../CSVfiles/balanced_mbti.csv")

def remove_types(text):
    # remove types names from the data
    types = ["istj", "isfj", "infj", "intj", "istp", "isfp", "infp", "intp",
             "estp", "esfp", "enfp", "entp", "estj", "esfj", "enfj", "entj"]
    # words that are 2 letters long to keep in the data
    two_letter_words = ['go', 'ex', 'k', 'ya', 'oh', 'ok', 'im', 'tv', 'yo', 'ah', 'pi', 'iq',
          'ft', 'en', 'hr', 'yr', 'ow', 'ew', 'ho', 'ai', 'eq', 'bi', 'th', 'st',
          'id', 'pb', 'hi', 'sc', 'um', 'uk', 'eu', 'pm', 'ha', 'kg', 'uh', 'bc',
          'hd', 'la', 'ad', 'pc', 'un', 'bf', 'em', 'xl', 'gf', 'pp', 'af', 'tf', 'dk',
          'cm', 'ui', 'pg', 'tm', 'dt', 'uw', 'ak', 'dm', 'km', 'cd', 'fr', 'ux',
          'vm', 'ip', 'mf', 'db', 'yu', 'α', 'us', 'df', 'qa', 'no']

    types +=[word + "s" for word in types]
    words = text.split()
    words = [w for w in words if not w in types]
    stops = set(stopwords.words('english'))

    words = [w for w in words if w not in stops and
             len(w) > 2 or w in two_letter_words]  # remove common words and very short words

    return ' '.join(words)

# mbti_data['posts'] = mbti_data['posts'].apply(remove_types) # clean the data set
# # save it in order to save run-time
# mbti_data.to_csv("../CSVfiles/mbti_no_types_names.csv",index=False)

# # run LR on the cleaned data
LR_model = LRModel("../CSVfiles/mbti_no_types_names.csv", cleaned=True)

LR_model.prediction()

# top_pos, top_neg = LR_model.get_feature_weights(top_n=30)
#
# print("Assertive words:")
# print(top_pos)
#
# print("\nNon-assertive words:")
# print(top_neg)

# run SVM on the cleaned data
# SVM_model = SVMModel("../CSVfiles/mbti_no_types_names.csv", cleaned=True)
#
# SVM_model.prediction()