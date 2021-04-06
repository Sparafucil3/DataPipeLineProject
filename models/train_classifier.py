# import nltk and parsing dictionaries
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# import libraries and functions
import sys
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    Loads SQL Lite DB from specified filepath, creates X & y date for ML processing
    
    Input: String with filepath
    Return X,y data for ML processing
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM CleanedMessages", engine)
    X = df['message']
    y = df.drop(columns=['message', 'id','original', 'genre'], axis=1)
    category_names=y.columns.values
    return(X,y, category_names)

def tokenize(text):
    '''
    Functions cleans, tokenizes, and lemmatized text and return the cleaned tokens
    Input: text string
    Outpur: cleaned list oject of tokens
    '''
    # delete punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ',text)
    # Tokenize 
    tokens = word_tokenize(text)
    # lemmatize
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for toke in tokens:
        clean_toke = lemmatizer.lemmatize(toke, pos='n').strip()
        clean_toke = lemmatizer.lemmatize(clean_toke, pos='v')
        cleaned_tokens.append(clean_toke)    
    
    return cleaned_tokens

def build_model():
    '''
    This function builds a modle, creates a pipeline with hyper-parameter tuning through
    use of a GridSearchCV and returns a said model
    
    Input: None
    Output: the best scoring model based on the parameters we are cycling through    
    '''
    # Create the pipeline for consistency in testing
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {'tfidf__norm': ['l1','l2'],
              'clf__estimator__criterion': ["gini", "entropy"],
              'clf__estimator__class_weight': ['balanced', None]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return(cv)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function evaluates the performance of a model provided
    Input: model, test data (X and Y) amd categories to be tested
    Output: None
    
    '''
    predicted = model.predict(X_test)
    print(classification_report(Y_test, predicted, target_names=category_names))
    return


def save_model(model, model_filepath):
    '''
    Serializes a model to disk
    Input: model to be serialized, string with a filepath to save the model to
    Output: None
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model (slowly)...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()