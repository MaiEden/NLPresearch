import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from Models.models_utilities import clean_text, predict_additional_posts, predict_post

class LRModel:

    def __init__(self, data_path, cleaned):
        """
        Initialize and train the Logistic Regression model.
        data_path - path to the data-set
        cleaned - boolean variable indicating whether the data-set is already cleaned
        """
        # read the data
        mbti_data = pd.read_csv(data_path)
        if not cleaned:
            mbti_data['posts'] = mbti_data['posts'].apply(clean_text)  # clean the data-set

        # convert text to vectors using TF-IDF
        # Prioritizes important words in the text and reduces common words
        # Uses only the 5000 most important words
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(mbti_data['posts'])

        # Creating a binary target variable: 1 - Assertive (ESTJ/ENTJ), 0 - Non-Assertive
        mbti_data['assertive'] = mbti_data['type'].apply(lambda x: 1 if x in ['ESTJ', 'ENTJ'] else 0)
        y = mbti_data['assertive']

        # Division into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creating a logistic regression model and training it
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train, y_train)
        self.model = logistic_model
        self.X_test = X_test
        self.y_test = y_test
        self.vectorizer = vectorizer

    def prediction(self):
        """
        Make predictions on the test set and print accuracy and classification report.
        """
        # Prediction on the test set
        logistic_predictions = self.model.predict(self.X_test)

        # print the results
        print("Logistic Regression Accuracy:", accuracy_score(self.y_test, logistic_predictions))
        print(classification_report(self.y_test, logistic_predictions))

    def predict_additional_posts(self, data_path, num_records=20):
        """
        Predict excluded posts using the trained model.
        data_path - path to the excluded posts.
        num_records - number of non-assertive records to retrieve from excluded data.
        """
        predict_additional_posts(self, data_path, num_records)

    def predict_post(self, post_text):
        """
        Predict a single post using the trained model.
        """
        return predict_post(post_text, self.vectorizer, self.model)

    def get_feature_weights(self, top_n=20):
        """
        Returns the words with the most assertive and non-assertive weights.
        """
        feature_names = self.vectorizer.get_feature_names_out()  # List of words
        coefs = self.model.coef_[0]  # Weight vector (binary – just one row)

        coef_df = pd.DataFrame({
            'word': feature_names,
            'weight': coefs
        })

        # Words that lean most towards assertiveness (label=1)
        top_assertive = coef_df.sort_values('weight', ascending=False).head(top_n)

        # Words that lean most towards non-assertive (label=0)
        top_non_assertive = coef_df.sort_values('weight', ascending=True).head(top_n)

        return top_assertive, top_non_assertive

    def get_word_weight(self, word):
        """
        Returns the weight of a given word (if it exists in the vocabulary)
        """
        feature_names = self.vectorizer.get_feature_names_out()
        # Create a mapping from word to its index
        word_to_idx = {w: i for i, w in enumerate(feature_names)}

        if word not in word_to_idx:
            return None  # The word did not appear in the corpus/was not output by the vectorizer

        idx = word_to_idx[word]
        weight = self.model.coef_[0][idx]
        return weight