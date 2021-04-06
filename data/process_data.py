import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' 
    This function reads the input files and mergers into a single df and returns that df
    Inputs: 
        messages_filepath: path to the messages file -- expects a common mapping key 'id'
        categories_filepath: path to categories file -- expects a common mapping key 'id'
    Returns: 
        Dataframe of both files merged on the common 'id' field
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return (pd.merge(messages, categories, on='id', how='left'))   


def clean_data(df):
    '''
    This function accepts a dataframe of merged messages and categories and prepares if for 
    Machine Learning. As part of this process:
        1) It splits the categories column into separate categories df
        2) Add column names to this new categories df
        3) Loops through each row of the categories df and replaces the value there with an 
            integer 1 or 0 depending on the existing value
        4) Drops the original "categoies" column from the input df
        5) Merges the input df and the new categories df dropping duplicates
        6) returns the cleaned df
        
    Input: dataframe of combined messages and categories
    Return: cleaned df ready for ML processing
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0].str.split('-', expand = True)
    category_colnames = list(row[0])
    categories.columns = category_colnames
    
    for column in categories.columns:
        # cast to string to make sure we are working with strings
        categories[column].apply(str)
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(-1)
        # convert column from string to int
        categories[column] = categories[column].astype(int)
    
    # ensure all values between 0 and 1
    categories = categories.clip(0,1)
    
    df.drop(columns=['categories'], inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1, join='inner').drop_duplicates()
    return(df)

def save_data(df, database_filename):
    '''
    Function saves the cleaned dataframe to a SQL-lite DB located at the input filename    
    Input: 
        DF to be put saved on disk as a DB
        string containing the filename/path to save the DB to
    Return: None
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('CleanedMessages', engine, index=False, if_exists = 'replace')
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()