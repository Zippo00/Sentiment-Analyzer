from flask import Flask, render_template, request
from turbo_flask import Turbo
import backend_functions
import json
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
turbo = Turbo(app)


#Placeholder figure. Remove later
placeholder_df = px.data.tips()
placeholder_fig = px.histogram(placeholder_df, x="day")


@app.route('/')
def index():
    #*Insert script here to plot some graph that will be shown first when user navigates to the web page*
    fig = placeholder_fig
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template('index.html', graphJSON=graphJSON)


@app.route('/plot_graph', methods=['POST'])
def plot_graph():
    # *Insert script to check posted variables, and plot the corresponding graph with plotly*
    dataset = request.form.get('dataset')
    graph_to_plot = request.form.get('graphToPlot')
    if dataset == 'hotel_reviews':
        if graph_to_plot == 'task3':
            fig = backend_functions.construct_histogram_for_star_categories('data/London_hotel_reviews.csv')
        elif graph_to_plot == 'task4':
            fig = backend_functions.proportion_of_positive_and_negative_subclass_in_ambiguous_class('data/London_hotel_reviews.csv','subclasses_table', 'D1.db')
        elif graph_to_plot == 'task5_1':
            fig1, fig2 = backend_functions.task5_plotly('data/London_hotel_reviews.csv')
            fig = fig1
        elif graph_to_plot == 'task5_2':
            fig1, fig2 = backend_functions.task5_plotly('data/London_hotel_reviews.csv')
            fig = fig2
        elif graph_to_plot == 'task11':
            fig = backend_functions.occurrence_of_positive_and_negative_words('data/London_hotel_reviews.csv')
    else:
        return json.dumps("This part of the application is not developed yet. Try with a different dataset.")
    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graphJSON


@app.route('/calculate', methods=['POST'])
def calculate():
    # *Insert script to check posted variables, and perform calculations based on them*
    result = None
    dataset = request.form.get('dataset')
    calculation = request.form.get('calculation')
    if dataset == 'hotel_reviews':
        if calculation =='task2':
            pearson = backend_functions.correlation_coefficient('data/London_hotel_reviews.csv', 'raw_sentiment_scores', 'raw_sentiment_scores.db')
            result = f"Pearson's Correlation Coefficient for the sentiment scores and user's ratings of the selected dataset is: {pearson[0]:.3f}"
        elif calculation =='task3':
             result = "Based on the calculated mean, standard deviation & kurtosis for the selected dataset:\n-Hotels with Low Standard Deviation tend to have relatively consistent ratings.\n\
-Hotels with High Standard Deviation tend to have more variable ratings."
        elif calculation =='task6':
            result = "A comparitively analysis on the results of LDA output with those generated from WordCloud indicate a substantial overlap between the high frequency words in the WordCloud and the \
top five words per topic in the positive and negative subclasses determined by LDA."
        elif calculation =='task8':
            pos_overlapping_ratio, neg_overlapping_ratio = backend_functions.task8()
            result = f'Comparing the Empath Categories generated for the reviews that belong to either "Positive" or "Negative" subclass, and the Empath Categories generated for Brown Reviews Corpus, \
the overlapping ratios were the following:\n\nRatio of Empath categories overlapping between "Brown Reviews Corpus" & "Positive Subclass Reviews": {pos_overlapping_ratio:.2f} %\nRatio of Empath categories \
overlapping between "Brown Reviews Corpus" & "Negative Subclass Reviews": {neg_overlapping_ratio:.2f} %\n\nThe logic used to consider categories to be overlapping: If the normalized weight for the category \
is over 0.001 in both, brown and positive/negative empaths, the category is considered to be overlapping.'
        elif calculation =='task9':
            pos_overlapping_ratio, neg_overlapping_ratio = backend_functions.task9()
            result = f'Comparing the Empath Categories generated for the reviews that belong to either "Positive" or "Negative" subclass, and the Empath Categories generated for Negative and Positive LDA Topics determined in Task 6, \
the overlapping ratios were the following:\n\nRatio of Empath categories overlapping between "Positive LDA Topics" & "Positive Subclass Reviews": {pos_overlapping_ratio:.2f} %\nRatio of Empath categories \
overlapping between "Negative LDA Topics" & "Negative Subclass Reviews": {neg_overlapping_ratio:.2f} %\n\nThe logic used to consider categories to be overlapping: First, all categories with a weighted value less than 0.1 were removed \
from the category sets for "Positive Subclass Reviews" and "Negative Subclass Reviews". Then the categories were considered to be overlapping, if the Empath category was found in the Empath category sets of both "Positive/Negative LDA Topics" \
and "Positive/Negative Subclass Reviews".\n\nNote: The overlapping ratio between "Negative LDA Topics" and "Negative Subclass Reviews" is 0.00%, due to the fact that the words/topics determined by LDA in Task 6 were not English words. Empath Client \
does not understand languages apart from English, at least at this time.'
        elif calculation =='task12':
            result = "Some of the associations can lead to evaluation errors if taken at face value. For example, is the pair 'cheapest - choice' really always a positive thing?\n\
Similarly, the adjectives 'adequate' & 'decent' could belong to either class.\n\
As per the common categories in task 11, 'room' was clearly the most commonly rated aspect, associated with multiple different adjectives."
    else:
        return json.dumps("This part of the application is not developed yet. Try with a different dataset.")
    if not result:
        result = "Pearson's Correlation for selected dataset is: 0.44"
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True)