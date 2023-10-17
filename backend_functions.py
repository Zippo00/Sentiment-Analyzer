'''
Functions for Review Sentiment Analyzer
'''
import sqlite3

import pandas as pd
from sentistrength import PySentiStr
import datahandling
from scipy import stats


def store_sent_score(csv_filepath, db_table, db):
    '''
    Calculate the SentiStrength Sentiment Scores for each ['Review Text'] data entry in the given .csv file.
    Stores all of the calculated scores into given database. 
    :param csv_filepath: (str) Filepath to .csv file to extract the data to analyze from.
    :param db_table: (str) Name of the table in the given database, to store the scores to.
    :param db: (str) Name of the database to store the scores to.
    '''
    sent_str = PySentiStr()
    sent_str.setSentiStrengthPath('F:/GitStuff/Sentiment-Analyzer/SentiStrength/SentiStrengthCom.jar') # Note: Provide absolute path instead of relative path. CHANGE THIS PATH BASED ON THE SYSTEM YOU ARE RUNNING THE FUNCTION ON.
    sent_str.setSentiStrengthLanguageFolderPath('F:/GitStuff/Sentiment-Analyzer/SentiStrength/SentStrength_Data_Sept2011/') # Note: Provide absolute path instead of relative path. CHANGE THIS PATH BASED ON THE SYSTEM YOU ARE RUNNING THE FUNCTION ON.
    df = pd.read_csv(csv_filepath, encoding = "ISO-8859-1")
    reviews = df['Review Text'].tolist()
    #Calculate sentiment scores for each review
    scores = sent_str.getSentiment(reviews, score='dual')
    #print(scores)
    query = f"INSERT INTO {db_table} VALUES "
    #Iterate through the scores & reviews, adding each (positive score, negative score, overall score) -data entry into given database
    for index, score in enumerate(scores):
        query += f"""({score[0]}, {score[1]}, {score[0]+score[1]}), """
    #Remove final comma and space from the query.
    query = query[:-2]
    #Execute query
    datahandling.sql_execute(query, db)

# correlation of the overall sentiment score of each review with the user’s rating
def correlation_coefficient(csv_filepath, db_table, db):
    scores = datahandling.fetch_data('raw_sentiment_score','raw_sentiment_score.db')

    overall_sentiment_score = [score[2] for score in scores]

    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    user_review_rating = df['Review Rating'].tolist()


    correlation_coefficient = stats.pearsonr(overall_sentiment_score, user_review_rating)
    print(correlation_coefficient)

    return correlation_coefficient


if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    pass
