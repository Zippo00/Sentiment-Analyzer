from flask import Flask, render_template, request
from turbo_flask import Turbo
import backend_functions
import json
import pandas as pd
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)
turbo = Turbo(app)

@app.route('/')
def index():
    #*Insert script here to plot some graph that will be shown first when user navigates to the web page*
    #fig = 
    #graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template('index.html')


@app.route('/plot_graph', methods=['POST'])
def plot_graph():
    # *Insert script to check posted variables, and plot the corresponding graph with plotly*
    #fig = 
    #graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    #return graphJSON
    pass

@app.route('/calculate', methods=['POST'])
def calculate():
    # *Insert script to check posted variables, and perform calculations based on them*
    result = "Pearson's Correlation for selected dataset is: 0.77"
    return result

if __name__ == '__main__':
    app.run(debug=True)