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
from empath import Empath
import nltk
from nltk import word_tokenize, pos_tag
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tree import *

nltk.download('averaged_perceptron_tagger') # POS tagging
nltk.download('stopwords') # stopwords

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

#Task 5
def task5(csv_filepath):
    (positive_reviews_text, negative_reviews_text) = classify_reviews(csv_filepath, stringify=True)

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

#Task 6
def proportion_of_positive_and_negative_subclass_in_ambiguous_class(csv_filepath):
    (positive_reviews, negative_reviews) = classify_reviews(csv_filepath, stringify=False)

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

def classify_reviews(csv_filepath, stringify=False):
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

    if stringify:
        # Concatenate all reviews for positive and negative subclasses  
        positive_reviews_text = ' '.join(positive_reviews['Review Text'])
        negative_reviews_text = ' '.join(negative_reviews['Review Text'])
        return (positive_reviews_text, negative_reviews_text)
    return (positive_reviews, negative_reviews)

def task7(csv_filepath):
    '''
    Generate categories with Empath Client for reviews of hotels that belong to
    the negative subclass or positive sublass. Store the generated results as json
    files into 'data' folder.
    :param csv_filepath: (string) Filepath to .csv file containing the review data.
    '''
    lexicon = Empath()
    subclass_table = datahandling.fetch_data('subclass_table', 'D1.db')
    subclasses = {}
    for i in subclass_table:
        subclasses[i[0]] = i[1]
    negative_subclass_reviews = []
    positive_subclass_reviews = []
    df = pd.read_csv('data/London_hotel_reviews.csv', encoding = "ISO-8859-1")
    for i in df.index:
        if subclasses[df['Property Name'][i]] == 'None':
            continue
        elif subclasses[df['Property Name'][i]] == 'Negative':
            negative_subclass_reviews.append(df['Review Text'][i])
        elif subclasses[df['Property Name'][i]] == 'Positive':
            positive_subclass_reviews.append(df['Review Text'][i])
    print(f'\nReviews of hotels in negative subclass: {len(negative_subclass_reviews)}')
    print(f'\nReviews of hotels in positive subclass: {len(positive_subclass_reviews)}')
    # Apply Empath Client to get categories for reviews
    neg_subclass_empath_cats = lexicon.analyze(negative_subclass_reviews, normalize=True)
    pos_subclass_empath_cats = lexicon.analyze(positive_subclass_reviews, normalize=True)
    # Remove any categories with a zero value
    to_remove = []
    for i in neg_subclass_empath_cats.items():
        if i[1] == 0.0:
            #print(i)
            to_remove.append(i[0])
    for i in to_remove:
        neg_subclass_empath_cats.pop(i)

    to_remove = []
    for i in pos_subclass_empath_cats.items():
        if i[1] == 0.0:
            #print(i)
            to_remove.append(i[0])
    for i in to_remove:
        pos_subclass_empath_cats.pop(i)
    # Store results as json files
    with open('data/neg_subclass_empath_cats.json', 'w') as f:
        json.dump(neg_subclass_empath_cats, f, sort_keys=True, indent=4)
    with open('data/pos_subclass_empath_cats.json', 'w') as f:
        json.dump(pos_subclass_empath_cats, f, sort_keys=True, indent=4)

# Task 11
def occurrence_of_positive_and_negative_words_in_ambiguous_class(csv_filepath):
    (positive_reviews_text, negative_reviews_text) = classify_reviews(csv_filepath, stringify=True)

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

# Task 12
def identify_nouns_for_positive_and_negative_adjectives_in_ambiguous_class(csv_filepath):
    (positive_reviews_text, negative_reviews_text) = classify_reviews(csv_filepath, stringify=True)

    def preprocess(text, remove_stopwords=False, stemming=False):
        words = word_tokenize(text)
        words = [w.lower() for w in words]
        if remove_stopwords:
            s_words = stopwords.words('english')
            words = [w for w in words if not w in s_words]
        return words

    positive_reviews_text_processed = preprocess(positive_reviews_text, remove_stopwords=True)
    negative_reviews_text_processed = preprocess(negative_reviews_text, remove_stopwords=True)

    positive_reviews_text_tagged = pos_tag(positive_reviews_text_processed)
    negative_reviews_text_tagged = pos_tag(negative_reviews_text_processed)

    def read_lexicon(filename):
        with open('data/hu_liu_lexicon/' + filename, 'r', errors='replace') as lexicon_file:
            words = [l.strip() for l in lexicon_file.readlines() if not l.startswith(';') and l.strip() != '']
        print(f'# of words in lexicon \'{filename}\': {len(words)}')
        return words

    # Consider only those lexicon words that are contained in the review texts
    lexicon_words_positive = [w for w in read_lexicon('positive-words.txt') if w in positive_reviews_text_processed]
    lexicon_words_negative = [w for w in read_lexicon('negative-words.txt') if w in negative_reviews_text_processed]
    
    def map_nouns_to_lexicon(tagged_tokens, lexicon_tokens):
        nouns_to_adjectives = {}
        for (lidx, lw) in enumerate(lexicon_tokens):
            for (ridx, (rw, tag)) in enumerate(tagged_tokens):
                if lw == rw:
                    # check a 2-word window around the word
                    for i in range(-2, 2):
                        # skip the lexicon word itself
                        if i == 0:
                            continue
                        # find a noun
                        (w, t) = tagged_tokens[ridx - i]
                        if t == 'NN':
                            if lw in nouns_to_adjectives.keys():
                                if w not in nouns_to_adjectives[lw]:
                                    nouns_to_adjectives[lw].append(w)
                            else:
                                nouns_to_adjectives[lw] = [w]
                            break # continue to next word after the first match
        return nouns_to_adjectives

    nouns_to_adjectives_positive = map_nouns_to_lexicon(positive_reviews_text_tagged, lexicon_words_positive)
    nouns_to_adjectives_negative = map_nouns_to_lexicon(negative_reviews_text_tagged, lexicon_words_negative)

    # print(nouns_to_adjectives_positive)
    # print(nouns_to_adjectives_negative)
    datahandling.sql_execute("DROP TABLE IF EXISTS task12;", 'task12.db')
    datahandling.sql_execute("""
    CREATE TABLE task12 (
        adj TEXT,
        nouns TEXT,
        UNIQUE(adj)
    );
    """, 'task12.db')
    query = f'INSERT INTO task12 (adj, nouns) VALUES '
    for (adj, noun_list) in nouns_to_adjectives_positive.items():
        print(json.dumps(noun_list))
        query += f"""('{adj}', '{json.dumps(noun_list)}'), """
    query = query[:-2]
    datahandling.sql_execute(query, 'task12.db')
    return datahandling.fetch_data('task12', 'task12.db', 'SELECT * FROM task12')

if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    # correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
    # group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis('data/London_hotel_reviews.csv')
    # construct_histogram_for_star_categories('data/London_hotel_reviews.csv')
    #proportion_of_positive_and_negative_subclass_in_ambiguous_class('data/London_hotel_reviews.csv','subclass_table', 'raw_sentiment_scores.db')
    #task5('data/London_hotel_reviews.csv')
    #identify_nouns_for_positive_and_negative_adjectives_in_ambiguous_class('data/London_hotel_reviews.csv')
    pass


