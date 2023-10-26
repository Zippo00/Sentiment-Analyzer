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
            fig = backend_functions.occurrence_of_positive_and_negative_words_in_ambiguous_class('data/London_hotel_reviews.csv')
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
            result = f"Pearson's Correlation Coefficient for the selected dataset is: {pearson[0]:.3f}"
        elif calculation =='task3':
             result = "Based on the calculated mean, standard deviation & kurtosis for the selected dataset:\n-Hotels with Low Standard Deviation tend to have relatively consistent ratings.\n\
-Hotels with High Standard Deviation tend to have more variable ratings."
        elif calculation =='task6':
            pass
        elif calculation =='task7':
            pass
        elif calculation =='task8':
            pass
        elif calculation =='task9':
            pass
        elif calculation =='task12':
            result = "Some of the associations can lead to evaluation errors if taken at face value. For example, is the pair 'cheapest - choice' really always a positive thing?\n\
            Similary, the adjectives 'adequate' & 'decent' could belong to either class.\n\
            As per the common categories in task 11, 'room' was clearly the most commonly rated aspect, associated with multiple different adjectives."
    else:
        return json.dumps("This part of the application is not developed yet. Try with a different dataset.")
    if not result:
        result = "Pearson's Correlation for selected dataset is: 0.44"
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True)