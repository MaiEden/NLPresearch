import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from dataProccesing.Models.models_utilities import clean_text, predict_additional_posts

class SVMModel:

    def __init__(self, data_path, cleaned):
        """
        Initialize and train the SVM model.
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

        # train SVM (Support Vector Machine)  model
        svm_model = SVC()
        svm_model.fit(X_train, y_train)

        self.model = svm_model
        self.X_test = X_test
        self.y_test = y_test
        self.vectorizer = vectorizer

    def prediction(self):
        """
        Make predictions on the test set and print accuracy and classification report.
        """
        # Prediction on the test set
        svm_predictions = self.model.predict(self.X_test)

        # print the results
        print("SVM Accuracy:", accuracy_score(self.y_test, svm_predictions))
        print(classification_report(self.y_test, svm_predictions))

    def predict_additional_posts(self, data_path, num_records=20):
        """
        Predict excluded posts using the trained model.
        data_path - path to the excluded posts.
        num_records - number of non-assertive records to retrieve from excluded data.
        """
        predict_additional_posts(self, data_path, num_records)