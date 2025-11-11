"""
Utility functions for model prediction and data processing.
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def retrieve_non_assertive_records(data_df, num_records):
    """
    Retrieve a DataFrame from excluded data with a balanced selection
     of non-assertive personality types.
    """
    # counts the num of records of each type
    other_counts = data_df['type'].value_counts()

    # create a new empty DataFrame
    selected_other_df = pd.DataFrame()

    # randomly chose records from each non-assertive type
    for personality_type in other_counts.index:
        type_df = data_df[data_df['type'] == personality_type]
        num_samples_for_type = min(num_records // 14 + 1, len(type_df))
        selected_other_df = pd.concat([selected_other_df, type_df.sample(n=num_samples_for_type, random_state=42)])

    # randomly chose num_records records to check prediction
    return selected_other_df.sample(n=num_records, random_state=42)


# function to predict on new text
def predict_post(post, vectorizer, model):
    """
    Predict whether a given post is assertive or non-assertive.
    """
    post_vector = vectorizer.transform([post])
    prediction = model.predict(post_vector)[0]
    return "assertive (1)" if prediction == 1 else "non-assertive (0)"


# function to clean text
def clean_text(text):
    """
    Clean the input text by removing links, punctuation,
    converting to lowercase, tokenizing, and removing stopwords.
    """
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation marks
    text = text.lower()  # convert to lowercase
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    stops = set(stopwords.words('english'))
    words = [w for w in words if w not in stops]  # remove common words
    return ' '.join(words)


def predict_additional_posts(model, data_path, num_records=20 ):
    """
    Predict excluded posts using the trained model.
    """
    excluded_df = pd.read_csv(data_path)

    print("\ntesting assertive posts:")

    success_count = 0

    assertive_df = excluded_df[(excluded_df['type'] == 'ENTJ') | (excluded_df['type'] == 'ESTJ')]
    for index, row in assertive_df.iterrows():
        post_text = row['posts']
        predicted_label = predict_post(post_text, model.vectorizer, model.model)
        print(f"ID: {index}, type: {row['type']}, prediction: {predicted_label}")
        if predicted_label == "assertive (1)":
            success_count += 1
    print(f"\nAssertive results: {success_count} successes out of {len(assertive_df)} tests\n")

    print("\ntesting non-assertive posts")
    # chose only non-assertive (not ENTJ and not ESTJ)
    other_df = excluded_df[(excluded_df['type'] != 'ENTJ') & (excluded_df['type'] != 'ESTJ')]

    non_assertive_df = retrieve_non_assertive_records(other_df, num_records)
    success_count = 0
    for index, row in non_assertive_df.iterrows():
        post_text = row['posts']
        predicted_label = predict_post(post_text, model.vectorizer, model.model)
        print(f"ID: {index}, type: {row['type']}, prediction: {predicted_label}")
        if predicted_label == "non-assertive (0)":
            success_count += 1
    print(f"\nAssertive results: {success_count} successes out of {len(non_assertive_df)} tests\n")