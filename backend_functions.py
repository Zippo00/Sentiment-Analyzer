'''
Functions for Review Sentiment Analyzer
'''
import sqlite3

import pandas as pd
from sentistrength import PySentiStr
import datahandling
import subprocess
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
    scores = datahandling.fetch_data(db_table,db)

    overall_sentiment_score = [score[2] for score in scores]

    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    user_review_rating = df['Review Rating'].tolist()


    correlation_coefficient = stats.pearsonr(overall_sentiment_score, user_review_rating)
    print(correlation_coefficient)

    return correlation_coefficient

def group_reviews_by_hotel(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding = "ISO-8859-1")
    # Group reviews by the hotel name and collect reviews in a list
    grouped_reviews = df.groupby('Property Name')['Review Text'].apply(list)
    sentiment_scores = {}
    for hotel_name, reviews in grouped_reviews.items():
        # Apply sentiment analysis to the list of reviews for each hotel
        sentiment_scores[hotel_name] = analyze_sentiment(reviews)



# This function should return sentiment scores
def analyze_sentiment(reviews):
    sentiment_scores = []

    # Path to the Sentistrength JAR file and the SentiStrength data folder
    sentistrength_jar = 'F:/GitStuff/Sentiment-Analyzer/SentiStrength/SentiStrengthCom.jar'
    sentistrength_data = 'F:/GitStuff/Sentiment-Analyzer/SentiStrength/SentStrength_Data_Sept2011/'

    for review in reviews:
        # we have to format the text appropriately for Sentistrength (e.g., remove special characters)
        # Then, call Sentistrength using subprocess
        command = ['java', '-jar', sentistrength_jar, 'sentidata', sentistrength_data, 'text', review]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Extract the sentiment score from the result
        sentiment_score = int(result.stdout.strip())

        sentiment_scores.append(sentiment_score)

    return sentiment_scores



if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    pass
