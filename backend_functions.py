'''
Functions for Review Sentiment Analyzer
'''
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls
from matplotlib import pyplot as plt
from scipy import stats, sparse
from scipy.stats import kurtosis
from sentistrength import PySentiStr
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from sklearn.feature_extraction.text import  CountVectorizer
from gensim import corpora, models
import datahandling
import spacy
import json
from empath import Empath
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, pos_tag_sents
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tree import *
import re

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

def plotly_wordcloud(text):
    '''
    Generates a Plotly WordCloud figure from the given text.
    :param text: (str) Text to generate WordCloud from.
    :return: (fig) Plotly figure.
    '''
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate(text)
    
    word_list=[]
    freq_list=[]
    fontsize_list=[]
    position_list=[]
    orientation_list=[]
    color_list=[]

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
        
    # get the positions
    x=[]
    y=[]
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
            
    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*100)
    new_freq_list
    
    trace = go.Scatter(x=x, 
                       y=y, 
                       textfont = dict(size=new_freq_list,
                                       color=color_list),
                       hoverinfo='text',
                       hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text',  
                       text=word_list
                      )
    
    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})
    
    fig = go.Figure(data=[trace], layout=layout)
    
    return fig

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
    fig = plt.figure(figsize=(5,4))
    plt.bar(review_ratings, proportions)
    plt.xlabel("Review Rating")
    plt.ylabel("Proportion of Hotels with High Standard Deviation")
    plt.title("Proportion of Hotels with High Standard Deviation by Review Rating")
    #plt.show()
    return tls.mpl_to_plotly(fig)

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
    fig = plt.figure(figsize=(5,4))
    plt.bar(subclass_counts.index, subclass_counts.values)
    plt.xlabel('Subclass')
    plt.ylabel('Count')
    plt.title('Proportion of Positive and Negative Subclasses in Ambiguous Class', fontsize=7)
    #plt.show()
    # Change the matplotlib figure into a Plotly figure and return it
    return tls.mpl_to_plotly(fig)

#Task 5
def concatenate_all_reviews_of_each_subclass_and_use_wordCloud_to_highlight_the_most_frequent_wording_used(csv_filepath):
    (positive_reviews_text, negative_reviews_text) = classify_reviews(csv_filepath, stringify=True)

    # WordCloud for the positive subclass
    positive_wordcloud = WordCloud(width=250, height=150, background_color='white').generate(positive_reviews_text)

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

def task5_plotly(csv_filepath):
    '''
    Same as task5 -function, but outputs the WordCloud figures as plotly figures,
    instead of matplotlib.
    :param csv_filepath: (str) Filepath to .csv file to extract the data to analyze from.
    :return: (fig) Returns two plotly figures.
    '''
    (positive_reviews_text, negative_reviews_text) = classify_reviews(csv_filepath, stringify=True)
    fig1 = plotly_wordcloud(positive_reviews_text)
    fig2 = plotly_wordcloud(negative_reviews_text)
    return fig1, fig2

#Task 6
def determine_the_topic_distribution_of_the_positive_and_negative_subclass(csv_filepath):
    nltk.download('punkt')
    nltk.download('stopwords')
    df = pd.read_csv(csv_filepath, encoding="ISO-8859-1")

    def preprocess_text(text):
        # Tokenize the text
        words = word_tokenize(text)

        # Remove punctuation and convert to lowercase
        words = [word.lower() for word in words if word.isalpha()]

        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]

        # Join the words back into a clean text
        cleaned_text = ' '.join(words)

        return cleaned_text

    # Apply the preprocessing function to the "Review Text" column
    df['Review Text'] = df['Review Text'].apply(preprocess_text)

    positive_df = df[df['Subclass'] == 'Positive']
    negative_df = df[df['Subclass'] == 'Negative']

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

    # Perform LDA for the positive subclass
    X_positive = vectorizer.fit_transform(positive_df['Review Text'])
    n_topics = 5  # Number of topics
    n_words = 5  # Number of words per topic
    lda_positive = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_positive.fit(X_positive)
    print(lda_positive)

    # Perform LDA for the negative subclass
    X_negative = vectorizer.fit_transform(negative_df['Review Text'])
    lda_negative = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_negative.fit(X_negative)

    # Extract the top n_words per topic for both positive and negative subclasses
    top_words_positive = []
    top_words_negative = []

    for topic_idx, topic in enumerate(lda_positive.components_):
        top_n_words_idx = topic.argsort()[-n_words:][::-1]
        top_words_for_topic = [vectorizer.get_feature_names_out()[i] for i in top_n_words_idx]
        top_words_positive.append(top_words_for_topic)

    print("Top positive words per topic:", top_words_positive)

    for topic_idx, topic in enumerate(lda_negative.components_):
        top_n_words_idx = topic.argsort()[-n_words:][::-1]
        top_words_for_topic = [vectorizer.get_feature_names_out()[i] for i in top_n_words_idx]
        top_words_negative.append(top_words_for_topic)

    print("Top negative words per topic:", top_words_negative)

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
def occurrence_of_positive_and_negative_words(csv_filepath):
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

    print(f'Reviews of hotels in negative subclass: {len(negative_subclass_reviews)}')
    print(f'Reviews of hotels in positive subclass: {len(positive_subclass_reviews)}')

    positive_reviews_text = ' '.join(negative_subclass_reviews)
    negative_reviews_text = ' '.join(positive_subclass_reviews)

    def preprocess(text):
        # remove pipes people use to separate sentences
        text = text.replace('|', '')

        # fix some individual character(s) noticed manually
        text = text.replace('\x92', "'")
        text = text.replace('\x94', '"')
        text = text.replace('\x96', 'û')

        # remove double spaces, add space after periods if missing
        # source: https://stackoverflow.com/a/29507362
        text = re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', text))

        # remove spaces around forward slashes
        text = re.sub(r'(?:(?<=\/) | (?=\/))','', text)

        # tokenize sentence words
        words = word_tokenize(text)

        return words

    cat_freq_pos = {}
    cat_freq_neg = {}

    with open('data/common_cat_ontology.json') as json_file:
        common_categories_ontology = json.load(json_file)
        common_categories = common_categories_ontology.keys()
        for c in common_categories:
            synset = common_categories_ontology[c]['synonyms']
            synset.extend(common_categories_ontology[c]['hypernyms'])
            synset.extend(common_categories_ontology[c]['hyponyms'])
            synset.append(c)
            fd_pos = FreqDist(token.lower() for token in preprocess(positive_reviews_text) if token.lower() in synset)
            fd_neg = FreqDist(token.lower() for token in preprocess(negative_reviews_text) if token.lower() in synset)
            cat_freq_pos[c] = fd_pos.N()
            cat_freq_neg[c] = fd_neg.N()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5,4))
    bar_pos = ax1.bar(cat_freq_pos.keys(), cat_freq_pos.values())
    bar_neg = ax2.bar(cat_freq_neg.keys(), cat_freq_neg.values())
    ax1.set_title('Category Occurrence (Positive Reviews)', fontsize=7)
    ax2.set_title('Category Occurrence (Negative Reviews)', fontsize=7)
    #plt.show()
    #Change matplotlib graph to plotly graph and return it
    return tls.mpl_to_plotly(fig)

# Task 12
def identify_nouns_for_positive_and_negative_adjectives(csv_filepath):
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

    print(f'Reviews of hotels in negative subclass: {len(negative_subclass_reviews)}')
    print(f'Reviews of hotels in positive subclass: {len(positive_subclass_reviews)}')

    positive_reviews_text = ' '.join(negative_subclass_reviews)
    negative_reviews_text = ' '.join(positive_subclass_reviews)

    def preprocess(text):
        print('Preprocessing...')

        # remove pipes people use to separate sentences
        text = text.replace('|', '')

        # fix some individual character(s) noticed manually
        text = text.replace('\x92', "'")
        text = text.replace('\x94', '"')
        text = text.replace('\x96', 'û')

        # remove double spaces, add space after periods if missing
        # source: https://stackoverflow.com/a/29507362
        text = re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', text))

        # remove spaces around forward slashes
        text = re.sub(r'(?:(?<=\/) | (?=\/))','', text)

        # tokenize sentences
        sentences = sent_tokenize(text)

        # tokenize sentence words
        sentences = [word_tokenize(sent) for sent in sentences]

        return sentences

    tokenized_sentences_pos = preprocess(positive_reviews_text)
    tokenized_sentences_neg = preprocess(negative_reviews_text)

    def read_lexicon(filename):
        with open('data/hu_liu_lexicon/' + filename, 'r', errors='replace') as lexicon_file:
            words = [l.strip() for l in lexicon_file.readlines() if not l.startswith(';') and l.strip() != '']
        print(f'# of words in lexicon \'{filename}\': {len(words)}')
        return words

    def read_and_filter_lexicon(filename, tokenized_sentences_filter):
        filtered_lexicon = []
        for w in read_lexicon(filename):
            for sent in tokenized_sentences_filter:
                # compare against lowercase words, keep casing in the original sentence
                sent_words = [w.lower() for w in sent]
                if w in sent_words:
                    filtered_lexicon.append(w)
        return filtered_lexicon
    
    # Consider only those lexicon words that are contained in the review texts
    print('Filtering lexicon...')
    lexicon_words_pos = read_and_filter_lexicon('positive-words.txt', tokenized_sentences_pos)
    lexicon_words_neg = read_and_filter_lexicon('negative-words.txt', tokenized_sentences_neg)

    # POS tag sentences before stopword removal to preserve sentence context
    print('POS tagging sentences')
    tokenized_sentences_pos = pos_tag_sents(tokenized_sentences_pos)
    tokenized_sentences_neg = pos_tag_sents(tokenized_sentences_neg)

    # remove stop words
    print('Removing stop words...')
    s_words = stopwords.words('english')
    def remove_stopwords(tokenized_tagged_sentences):    
        sentences_result = []
        for sent in tokenized_tagged_sentences:
            sentence_result = []
            for (w, tag) in sent:
                if w.lower() not in s_words: # lowercase comparison
                    sentence_result.append((w, tag))
            sentences_result.append(sentence_result)
        return sentences_result

    tokenized_sentences_pos = remove_stopwords(tokenized_sentences_pos)
    tokenized_sentences_neg = remove_stopwords(tokenized_sentences_neg)
    
    def map_nouns_to_lexicon(tagged_sentences, lexicon_tokens):
        nouns_to_adjectives = {}
        for (lidx, lw) in enumerate(lexicon_tokens):
            for (ridx, sent) in enumerate(tagged_sentences):
                for (sidx, (rw, tag)) in enumerate(sent):
                    if lw == rw.lower(): # lowercase comparison
                        # check a 2-word window around the word
                        for i in range(-2, 2):
                            # skip the lexicon word itself
                            if i == 0:
                                continue
                            # find a noun
                            try:
                                (w, t) = sent[sidx - i]
                                if t == 'NN':
                                    if lw in nouns_to_adjectives.keys():
                                        if w not in nouns_to_adjectives[lw]:
                                            nouns_to_adjectives[lw].append(w)
                                    else:
                                        nouns_to_adjectives[lw] = [w]
                                    break # continue to next word after the first match
                            except IndexError:
                                pass
        return nouns_to_adjectives

    print('Mapping nouns to lexicon...')
    nouns_to_adjectives_positive = map_nouns_to_lexicon(tokenized_sentences_pos, lexicon_words_pos)
    nouns_to_adjectives_negative = map_nouns_to_lexicon(tokenized_sentences_neg, lexicon_words_neg)

    db_name = 'task12.db'
    def store(data, table):
        datahandling.sql_execute(f"DROP TABLE IF EXISTS {table}", db_name)
        datahandling.sql_execute(f"""
        CREATE TABLE {table} (
            adj TEXT,
            nouns TEXT,
            UNIQUE(adj)
        )
        """, db_name)
        query = f"INSERT INTO {table} (adj, nouns) VALUES "
        for (adj, noun_list) in data.items():
            query += f"""('{adj}', '{json.dumps(noun_list, ensure_ascii=False).replace("'", "`")}'), """
        query = query[:-2]
        datahandling.sql_execute(query, db_name)
        data = datahandling.fetch_data(table, db_name, f"SELECT * FROM {table}")
        data_dict = {}
        for (adj, nouns) in data:
            data_dict[adj] = json.loads(nouns.replace("`", "'"))
        return data_dict
    
    print('Storing data...')
    data_pos = store(nouns_to_adjectives_positive, 'task12_pos')
    data_neg = store(nouns_to_adjectives_negative, 'task12_neg')

    # print('\nPOSITIVE:')
    # print(data_pos)
    # print('\nNEGATIVE:')
    # print(data_neg)

    return (data_pos, data_neg)

if __name__ == '__main__':
    # store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db') # I used this line to calculate and store all of the sentiment scores into raw_sentiment_scores.db database.
    # correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
    # group_reviews_by_hotel_and_calculate_mean_standard_deviation_and_kurtosis('data/London_hotel_reviews.csv')
    # construct_histogram_for_star_categories('data/London_hotel_reviews.csv')
    #proportion_of_positive_and_negative_subclass_in_ambiguous_class('data/London_hotel_reviews.csv','subclass_table', 'raw_sentiment_scores.db')
    #task5('data/London_hotel_reviews.csv')
    #occurrence_of_positive_and_negative_words('data/London_hotel_reviews.csv')
    #concatenate_all_reviews_of_each_subclass_and_use_wordCloud_to_highlight_the_most_frequent_wording_used('data/London_hotel_reviews.csv')
    #determine_the_topic_distribution_of_the_positive_and_negative_subclass('data/London_hotel_reviews.csv')
    #identify_nouns_for_positive_and_negative_adjectives_in_ambiguous_class('data/London_hotel_reviews.csv')
    pass


