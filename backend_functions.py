'''
Functions for Review Sentiment Analyzer
'''
import sqlite3

import pandas as pd
from sentistrength import PySentiStr
import datahandling
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis



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
    return correlation_coefficient

def group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding = "ISO-8859-1")
    grouped = df.groupby('Property Name')['Review Rating']
    results = grouped.agg(['mean', 'std', kurtosis]).reset_index()
    # Access and print the mean and standard deviation for each hotel
    for index, row in results.iterrows():
        hotel_name = row['Property Name']
        mean = row['mean']
        std = row['std']
        kurt = row['kurtosis']

        print(f'Hotel: {hotel_name}, Mean: {mean}, Std: {std}, Kurtosis: {kurt}')

    # Set a threshold to distinguish low and high standard deviations
    std_deviation_threshold = 1.0  # You can adjust this threshold as needed

    # Identify hotels with low and high standard deviations
    low_std_deviation_hotels = results[results['std'] < std_deviation_threshold]
    high_std_deviation_hotels = results[results['std'] >= std_deviation_threshold]
    print("Hotels with Low Standard Deviation:")
    print(low_std_deviation_hotels)
    print("\nHotels with High Standard Deviation:")
    print(high_std_deviation_hotels)

    # Comment whether the high variation of standard deviation occurs in expensive hotel or cheap hotels
    print("Hotels with Low Standard Deviation tend to have relatively consistent ratings.")
    print("Hotels with High Standard Deviation tend to have more variable ratings.")


def construct_histogram_for_star_categories(csv_filepath):
    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")

    # list to store the proportion of hotels exceeding the threshold for each review rating
    review_ratings = df['Review Rating'].unique()
    proportions = []
    std_deviation_threshold = 0.5 # You can adjust this threshold as needed

    for rating in review_ratings:
        subset = df[df['Review Rating'] == rating]
        print('subset',subset)
        std_deviation_subset = subset.groupby('Property Name')['Review Rating'].std()
        proportion = (std_deviation_subset > std_deviation_threshold).mean()
        proportions.append(proportion)

    # Create a histogram to visualize proportions for each review rating
    plt.bar(review_ratings, proportions)
    plt.xlabel("Review Rating")
    plt.ylabel("Proportion of Hotels with High Standard Deviation")
    plt.title("Proportion of Hotels with High Standard Deviation by Review Rating")
    plt.show()


if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
    group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis('data/London_hotel_reviews.csv')
    construct_histogram_for_star_categories('data/London_hotel_reviews.csv')


