'''
Functions for Review Sentiment Analyzer
'''
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from scipy import stats, sparse
from scipy.stats import kurtosis
from sentistrength import PySentiStr
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import datahandling
import spacy
import json
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

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

# Task 2
# correlation of the overall sentiment score of each review with the user’s rating
def correlation_coefficient(csv_filepath, db_table, db):
    scores = datahandling.fetch_data(db_table,db)
    overall_sentiment_score = [score[2] for score in scores]
    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    user_review_rating = df['Review Rating'].tolist()
    correlation_coefficient = stats.pearsonr(overall_sentiment_score, user_review_rating)
    return correlation_coefficient

# Task 3
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

        #print(f'Hotel: {hotel_name}, Mean: {mean}, Std: {std}, Kurtosis: {kurt}')

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
        #print('subset',subset)
        std_deviation_subset = subset.groupby('Property Name')['Review Rating'].std()
        proportion = (std_deviation_subset > std_deviation_threshold).mean()
        proportions.append(proportion)

    # histogram to visualize proportions for each review rating
    plt.bar(review_ratings, proportions)
    plt.xlabel("Review Rating")
    plt.ylabel("Proportion of Hotels with High Standard Deviation")
    plt.title("Proportion of Hotels with High Standard Deviation by Review Rating")
    plt.show()


# Task 4
def proportion_of_positive_and_negative_subclass_in_ambiguous_class(csv_filepath, db_table, db):

    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")
    std_deviation_threshold = 1.0
    hotel_stats = df.groupby('Property Name')['Review Rating'].agg(['std', 'mean']).reset_index()
    print(hotel_stats)
    print(hotel_stats['Property Name'])

    def determine_subclass(row):
        if row['std'] > std_deviation_threshold:
            # If the standard deviation is above the threshold, check the sentiment
            if row['mean'] > 3.5:
                return 'Positive'
            else:
                return 'Negative'
        else:
            return None

    hotel_stats['Subclass'] = hotel_stats.apply(determine_subclass, axis=1)
    print(hotel_stats)

    queryToCreateTable = f"""CREATE TABLE IF NOT EXISTS {db_table} (property_name text PRIMARY KEY, sub_class text NOT NULL)"""
    insert_query = f"INSERT INTO {db_table} VALUES"
    for propertyName, subClass in zip(hotel_stats['Property Name'], hotel_stats['Subclass']):
        insert_query += f"""('{propertyName}', '{subClass}'),"""
    insert_query = insert_query[:-1]  # Remove the trailing comma

    datahandling.sql_execute(queryToCreateTable,db)
    datahandling.sql_execute(insert_query,db)

    # Load the data from the database
    query = f"""SELECT sub_class FROM {db_table} WHERE sub_class IS NOT NULL"""
    subclass_data = datahandling.fetch_data(db_table, db,query)

    if not isinstance(subclass_data, pd.DataFrame):
        subclass_data = pd.DataFrame(subclass_data, columns=["sub_class"])

    # Count the occurrences of each subclass
    subclass_counts = subclass_data['sub_class'].value_counts()
    print('Subclass!!!!!!!!!!!!!!!!',subclass_counts)

    # Plot the histogram
    plt.bar(subclass_counts.index, subclass_counts.values)
    plt.xlabel('Subclass')
    plt.ylabel('Count')
    plt.title('Proportion of Positive and Negative Subclasses in Ambiguous Class')
    plt.show()

Task 5
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

Task 6
def proportion_of_positive_and_negative_subclass_in_ambiguous_class(csv_filepath):
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

    # Create a DataFrame for each subclass
    positive_df = pd.DataFrame({'Review': positive_reviews, 'Sentiment': 'positive'})
    negative_df = pd.DataFrame({'Review': negative_reviews, 'Sentiment': 'negative'})
    # Combine the data into a single DataFrame
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

    # Preprocess the text data using spaCy
    nlp = spacy.load("en_core_web_sm")
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_punct]
        return " ".join(tokens)

    combined_df['Preprocessed Review'] = combined_df['Review'].apply(preprocess_text)

    #  Initialize a TF-IDF vectorizer and Vectorize the Text
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')

    # Fit and transform the preprocessed text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['Preprocessed Review'])
    sparse.save_npz('tfidf_matrix.npz', sparse.csr_matrix(tfidf_matrix))

    tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')

    # Convert the TF-IDF matrix to a Gensim corpus
    corpus = corpora.MmCorpus(sparse.csr_matrix(tfidf_matrix))

    # Apply LDA
    num_topics = 5  # Number of topics
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=tfidf_vectorizer.get_feature_names_out(),
                                passes=15)

    # use the lda_model and corpus to get topic distributions for each review, Store this information in your database D1

    # Store the topic distribution for each review
    topic_distributions = [lda_model[review] for review in corpus]
    print(topic_distributions)

    for i, distribution in enumerate(topic_distributions):
        datahandling.sql_execute("INSERT INTO D1 (review_id, topic_distribution) VALUES (%s, %s)", (i, distribution))

    # Compare the LDA results with WordCloud findings for overlaps and relevance.
    #not able to import wordcloud and gensim

def task11(csv_filepath):
    # TODO copy paste, extract a func
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

    # New for task 11
    cat_freq_pos = {}
    cat_freq_neg = {}

    with open('data/common_cat_ontology.json') as json_file:
        common_categories_ontology = json.load(json_file)
        common_categories = common_categories_ontology.keys()
        for c in common_categories:
            synset = common_categories_ontology[c]['synonyms']
            synset.append(c)
            fd_pos = FreqDist(token.lower() for token in word_tokenize(positive_reviews_text) if token.lower() in synset)
            fd_neg = FreqDist(token.lower() for token in word_tokenize(negative_reviews_text) if token.lower() in synset)
            cat_freq_pos[c] = fd_pos.N()
            cat_freq_neg[c] = fd_neg.N()
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    bar_pos = ax1.bar(cat_freq_pos.keys(), cat_freq_pos.values())
    bar_neg = ax2.bar(cat_freq_neg.keys(), cat_freq_neg.values())
    ax1.set_title('category occurrence (positive reviews)')
    ax2.set_title('category occurrence (negative reviews)')
    plt.show()

if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    # correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
    # group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis('data/London_hotel_reviews.csv')
    # construct_histogram_for_star_categories('data/London_hotel_reviews.csv')
    proportion_of_positive_and_negative_subclass_in_ambiguous_class('data/London_hotel_reviews.csv','subclass_table', 'raw_sentiment_scores.db')
    #task5('data/London_hotel_reviews.csv')
    #task11('data/London_hotel_reviews.csv')


