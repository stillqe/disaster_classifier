import sys

import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

SEED = 123


def load_data(database_filepath):
    """Load data from database"""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster', engine)

    X = df.message.values
    y = df.loc[:, 'related':].values
    categories = list(df.loc[:, 'related':].columns.values)

    return X, y, categories


def tokenize(text):
    """Regularize, Lemmatize, Remove stop words, and return tokens"""
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    """Build machine learning pipeline consisting of CounterVectorizer, TfidfTransformer,
    and multi-class Classifier with Random Forest model"""
    pipeline = Pipeline([('count_vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=SEED)))])

    parameters = {
        'count_vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__bootstrap': [True, False]
    }

    cv = GridSearchCV(pipeline, parameters, cv=2)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """evaluate models for each category"""
    total_acc = 0
    y_pred = model.predict(X_test)

    for i in range(y_test.shape[1]):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        total_acc = total_acc + acc
        print(category_names[i])
        print(classification_report(y_test[:, i], y_pred[:, i]))

    print('**********************************')
    print('\nAverage Accuracy: {}'.format(total_acc / y_test.shape[1]))


def save_model(model, model_filepath):
    """Save model with the best parameter"""
    import pickle

    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def run_pipeline(database_filepath, model_filepath):
    """Run pipeline from loading data, training model to saving trained model"""
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        run_pipeline(database_filepath, model_filepath)
    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
