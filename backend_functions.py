'''
Functions for Review Sentiment Analyzer
'''
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import kurtosis
from sentistrength import PySentiStr
from wordcloud import WordCloud
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

    # threshold to distinguish low and high standard deviations
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

    # histogram to visualize proportions for each review rating
    plt.bar(review_ratings, proportions)
    plt.xlabel("Review Rating")
    plt.ylabel("Proportion of Hotels with High Standard Deviation")
    plt.title("Proportion of Hotels with High Standard Deviation by Review Rating")
    plt.show()

def proportion_of_positive_and_negative_subclass_in_ambiguous_class(csv_filepath):

    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    std_deviation_threshold = 1.0

    hotel_stats = df.groupby('Property Name')['Review Rating'].std()
    ambiguous_class_hotels = hotel_stats[hotel_stats > std_deviation_threshold].index
    print('ambi"""""""', ambiguous_class_hotels)

    # list to store classification results
    classification_results = []

    for hotel in ambiguous_class_hotels:
        hotel_reviews = df[df['Property Name'] == hotel]
        positive_reviews = hotel_reviews[
            hotel_reviews['Review Rating'] >= 4]  # Example: Consider ratings of 4 and 5 as positive
        negative_reviews = hotel_reviews[
            hotel_reviews['Review Rating'] <= 2]  # Example: Consider ratings of 1 and 2 as negative

        if len(positive_reviews) > len(negative_reviews):
            subclass = 'Positive'
            print('Positive"""""""', subclass)
        else:
            subclass = 'Negative'

        classification_results.append({'Hotel': hotel, 'Ambiguous': 'Yes', 'Subclass': subclass})

    # DataFrame to store the classification results
    D1 = pd.DataFrame(classification_results)

    # histogram to visualize the proportion of positive and negative subclasses in the Ambiguous Class
    subclass_proportions = D1['Subclass'].value_counts()
    subclass_proportions.plot(kind='bar')

    plt.xlabel("Subclass")
    plt.ylabel("Proportion of Hotels")
    plt.title("Proportion of Positive and Negative Subclasses in the Ambiguous Class")
    plt.show()

def task5(csv_filepath):


    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    std_deviation_threshold = 1.0

    hotel_stats = df.groupby('Property Name')['Review Rating'].std()
    ambiguous_class_hotels = hotel_stats[hotel_stats > std_deviation_threshold].index


    for hotel in ambiguous_class_hotels:
        hotel_reviews = df[df['Property Name'] == hotel]
        positive_reviews = hotel_reviews[
            hotel_reviews['Review Rating'] >= 4]  # Example: Consider ratings of 4 and 5 as positive
        negative_reviews = hotel_reviews[
            hotel_reviews['Review Rating'] <= 2]  # Example: Consider ratings of 1 and 2 as negative

    # Concatenate all reviews for positive and negative subclasses
    positive_reviews_text = ' '.join(positive_reviews['Review Text'])
    negative_reviews_text = ' '.join(negative_reviews['Review Text'])

    # WordCloud for the positive subclass
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews_text)

    # WordCloud for the negative subclass
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_reviews_text)

    # WordCloud for the positive subclass
    plt.figure(figsize=(10, 5))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('WordCloud for Positive Subclass')
    plt.axis('off')
    plt.show()

    # WordCloud for the negative subclass
    plt.figure(figsize=(10, 5))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('WordCloud for Negative Subclass')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    # correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
    # group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis('data/London_hotel_reviews.csv')
    # construct_histogram_for_star_categories('data/London_hotel_reviews.csv')
    #proportion_of_positive_and_negative_subclass_in_ambiguous_class('data/London_hotel_reviews.csv')
    task5('data/London_hotel_reviews.csv')


