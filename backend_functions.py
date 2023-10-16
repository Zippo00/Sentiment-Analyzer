'''
Functions for Review Sentiment Analyzer
'''
import pandas as pd
from sentistrength import PySentiStr
import datahandling


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
    #Iterate through the scores & reviews, adding each (review, positive score, negative score, overall score) -data entry into given database
    for index, score in enumerate(scores):
        #print(f'TEXT:\n{text}\nSENTIMENT SCORE: {score}')
        #print(f'({type(reviews[index])}, {type(score[0])}, {type(score[1])}, {type(score[0]+score[1])})')
        query += f"""({score[0]}, {score[1]}, {score[0]+score[1]}), """
    #Remove final comma and space from the query.
    query = query[:-2]
    datahandling.sql_execute(query, db)


if __name__ == '__main__':
    #store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.