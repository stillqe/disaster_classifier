import pandas as pd
from plotly.graph_objs import Bar, Pie
import sys
sys.path.append('../')
from models.train_classifier import tokenize
from sklearn.feature_extraction.text import CountVectorizer

def get_figures(df):
    """Creates plotly visualizations

    Args:
        df: dataframe of disaster response

    Returns:
        list (dict): list containing the plotly visualizations

    """

    sorted_by_sum = df.loc[:, 'related':].sum().sort_values(ascending=False)

    figure1 = {
        'data': [
            Bar(
                x=sorted_by_sum.index.tolist(),
                y=sorted_by_sum
            )],

        'layout': {
            'title': 'Category Outlook',
            'xaxis':{
                'title': 'Categories'
            },
            'yaxis': {
                'title': 'Number of messages'
            }
        }
    }

    multi_label_counts = df.loc[:, 'related':].sum(axis=1).value_counts()

    figure2 = {
        'data':[
            Bar(
                x= multi_label_counts.index,
                y=multi_label_counts.values
            )
        ],
        'layout': {
            'title': 'How many labels the trained messages have?',
            'xaxis': {
                'title': 'Number of labels'
            },
            'yaxis':{
                'title': 'Number of messages'
            }
        }
    }


    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    figure3 = {
        'data': [
            Pie(
                labels=genre_names,
                values=genre_counts
            )],

        'layout': {
            'title': 'Distribution of Message Genres'
        }
    }

    top_50 = get_top_ngram(df.message)
    x, y = map(list, zip(*top_50))

    figure4 = {
        'data': [
            Bar(
                x=x,
                y=y
            )
        ],

        'layout': {
            'title': 'Top 50 Common Words'
        }
    }

    return [figure1, figure2, figure3, figure4]


def get_top_ngram(corpus, top=50, n=1):
    vect = CountVectorizer(tokenizer=tokenize, ngram_range=(1,1), analyzer='word')
    bow = vect.fit_transform(corpus)
    sum_words = bow.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]