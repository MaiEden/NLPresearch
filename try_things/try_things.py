from tokenize import String
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from dataProccesing.Models.LR_model import LRModel
from dataProccesing.Models.SVM_model import SVMModel

LR_model = LRModel("../CSVfiles/balanced_mbti.csv", cleaned=True)

#===========================================================================
# experiment on populations of people who are usually not assertive
# ==========================================================================
#
# new_df = pd.read_csv("cleaned_posts.csv")
#
# # assume all records are non-assertive
# success_count = 0
# failure_count = 0
#
# for index, row in new_df.iterrows():
#     post_text = row['cleaned_body']
#     if not isinstance(post_text, str):
#         continue
#     predicted_label = LR_model.predict_post(post_text)
#     if predicted_label == "non-assertive (0)":
#         success_count += 1
#     else:
#         failure_count += 1
#
# total = success_count + failure_count
# accuracy = (success_count / total) * 100 if total > 0 else 0
#
# print("\n--- Model results on new file ---")
# print(f"Total records: {total}")
# print(f"Successes (non-assertive): {success_count}")
# print(f"Failures (assertive): {failure_count}")
# print(f"Success percentage: {accuracy:.2f}%")

#===============================================
# experiment on women and mans populations
#===============================================
#
# new_df = pd.read_csv("cleaned_posts2.csv")
#
# # gender counts
# total_counts = {"all": 0, "male": 0, "female": 0}
# assertive_counts = {"all": 0, "male": 0, "female": 0}
#
# for index, row in new_df.iterrows():
#     post_text = row['cleaned_body']
#     gender = row.get('g', None)
#
#     if not isinstance(post_text, str) or gender not in [0, 1]:
#         continue
#
#     predicted_label = LR_model.predict_post(post_text)
#
#     # counts updates
#     total_counts["all"] += 1
#     if gender == 1:
#         total_counts["male"] += 1
#     elif gender == 0:
#         total_counts["female"] += 1
#
#     # assertive counts updates
#     if predicted_label == "non-assertive (0)":
#         continue
#     assertive_counts["all"] += 1
#     if gender == 1:
#         assertive_counts["male"] += 1
#     elif gender == 0:
#         assertive_counts["female"] += 1
#
# # percentage calculation
# def calc_percent(part, whole):
#     return (part / whole) * 100 if whole > 0 else 0
#
# print("\n--- Assertiveness Analysis by Gender ---")
# print(f"Total Posts: {total_counts['all']}")
# print(f"Assertives in the Total Population: {assertive_counts['all']} ({calc_percent(assertive_counts['all'], total_counts['all']):.2f}%)")
# print(f"\nAmong Women (g=0):")
# print(f"Total: {total_counts['female']}")
# print(f"Assertiveness by Model: {assertive_counts['female']} ({calc_percent(assertive_counts['female'], total_counts['female']):.2f}%)")
# print(f"\nAmong Men (g=1):")
# print(f"Total: {total_counts['male']}")
# print(f"Assertives by Model: {assertive_counts['male']} ({calc_percent(assertive_counts['male'], total_counts['male']):.2f}%)")

#==========================================
# experiment on lectures at the UN
# =========================================
#
# new_df = pd.read_csv("cleaned_posts3.csv")
#
# # assume all records are assertive
# non_assertive_count = 0
# assertive_count = 0
#
# for index, row in new_df.iterrows():
#     post_text = row['cleaned_body']
#     if not isinstance(post_text, str):
#         continue
#     predicted_label = LR_model.predict_post(post_text)
#     if predicted_label == "non-assertive (0)":
#         non_assertive_count += 1
#     else:
#         assertive_count += 1
#
# total = non_assertive_count + assertive_count
# accuracy = (assertive_count / total) * 100 if total > 0 else 0
#
# print("\n--- Model results on new file ---")
# print(f"Total records: {total}")
# print(f"Non-assertive: {non_assertive_count}")
# print(f"Assertive: {assertive_count}")
# print(f"Assertive percentage: {accuracy:.2f}%")

#=================================================
# experiment on not clean MBTI data from Twitter
# ================================================
# def clean_text_not_rmv_stopwords(text):
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.lower()
#     tokens = word_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]
#     return ' '.join(words)
#
# def predict_post_not_clean(post, vectorizer, model):
#     # clen the new text
#     post = clean_text_not_rmv_stopwords(post)
#     post_vector = vectorizer.transform([post])
#     prediction = model.predict(post_vector)[0]
#
#     return "assertive (1)" if prediction == 1 else "non-assertive (0)"
#
# new_df = pd.read_csv("twitter_MBTI.csv")
#
# success_count = 0
# failed_count = 0
#
# for index, row in new_df.iterrows():
#     post_text = row['text']
#     if not isinstance(post_text, str):
#         continue
#     predicted_label = predict_post_not_clean(post_text, LR_model.vectorizer, LR_model.model)
#     if predicted_label == "non-assertive (0)" and row['label'] != "ENTJ" and row['label'] != "ESTJ":
#         success_count += 1
#     elif predicted_label == "assertive (1)" and row['label'] == "ENTJ" and row['label'] == "ESTJ":
#         success_count += 1
#     else:
#         failed_count += 1
#
# total = success_count + failed_count
# accuracy = (success_count / total) * 100 if total > 0 else 0
#
# print("\n--- Model results on new file ---")
# print(f"Total records: {total}")
# print(f"Success: {success_count}")
# print(f"Failed: {failed_count}")
# print(f"Success percentage: {accuracy:.2f}%")

#=======================================
# experiment no not cleaned data-set
# ======================================
#with MBTI types name
SVM_model = SVMModel("PersonalityCafe_mbti.csv", cleaned=True)

SVM_model.prediction()

mbti_data = pd.read_csv("PersonalityCafe_mbti.csv")

def remove_types(text):
    # remove MBTI type names from the text
    types = ["istj", "isfj", "infj", "intj", "istp", "isfp", "infp", "intp",
             "estp", "esfp", "enfp", "entp", "estj", "esfj", "enfj", "entj"]
    words = text.split()
    words = [w for w in words if not w in types]
    return ' '.join(words)

# mbti_data['posts'] = mbti_data['posts'].apply(remove_types)
# mbti_data.to_csv('PersonalityCafe_no_types_names.csv', index=False)

#without MBTI types name
SVM_model = SVMModel("PersonalityCafe_no_types_names.csv", cleaned=True)

SVM_model.prediction()