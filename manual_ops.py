'''
For managing the database etc.

Write the script you wish to execute after line: "if __name__ == _main_", save the file and execute via terminal.
'''
import datahandling


if __name__ == '__main__':
    #datahandling.sql_execute("CREATE TABLE raw_sentiment_scores (pos_score INTEGER, neg_score INTEGER, overall_score INTEGER)", "raw_sentiment_scores.db") # I used this for creating the existing .db file -Mikko