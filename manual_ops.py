'''
For managing the database etc.

Write the script you wish to execute after line: "if __name__ == _main_", save the file and execute via terminal.
'''
import datahandling
import backend_functions


if __name__ == '__main__':
    #datahandling.sql_execute("CREATE TABLE raw_sentiment_scores (pos_score INTEGER, neg_score INTEGER, overall_score INTEGER)", "raw_sentiment_scores.db") # I used this for creating the existing raw_sentiment_scores.db file -Mikko

    #These 2 lines were used to create the D1.db file with two tables: 'raw_sentiment_scores' and 'ambitious_classes'
    #datahandling.sql_execute("CREATE TABLE raw_sentiment_scores (pos_score INTEGER, neg_score INTEGER, overall_score INTEGER)", "D1.db")
    #datahandling.sql_execute("CREATE TABLE ambitious_classes (ambitious_class TEXT)", "D1.db")

    #Then this line was used to calculate the sentiment scores, and store them into the D1.db table named 'raw_sentiment_scores':
    #backend_functions.store_sent_score('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'D1.db')
    pass
