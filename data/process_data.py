import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from csv file into dataframe and return"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets|
    df = pd.merge(messages, categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """wrangle raw dataframe and return cleaned dataframe"""
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda str: str[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop(labels=['categories'], axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Save cleaned dataframe to SQL database"""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster', engine, index=False, if_exists='replace')


def run_etl_pipeline(messages_filepath, categories_filepath, database_filepath):
    """Run ETL pipeline from loading data, cleaning data, and saving data"""
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')




if __name__ == '__main__':
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        run_etl_pipeline(messages_filepath, categories_filepath, database_filepath)
    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')
