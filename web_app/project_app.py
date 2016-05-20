import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from flask import Blueprint, request, render_template

#---------- MODEL IN MEMORY ----------------#
thresh = .5

class Dict_obj(object):
    stats_dict = {}

data_log_reg = pd.read_csv("data/data_logReg.csv")
data_rand_forest = pd.read_csv("data/data_randForest.csv")

data = data_rand_forest

status = data['status']
pred_func = data['func'].values

#lat = data['latitude']
#lon = data['longitude']

prediction = np.array([1 if x>thresh else 0 for x in pred_func])
not_prediction = [1 if x==0 else 0 for x in prediction]
actual = np.array([1 if x==2 else 0 for x in status])
not_actual = [1 if x==0 else 0 for x in actual]

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    data_rand_forest = pd.read_csv("data/data_randForest.csv")
    data = data_rand_forest
    table_html = (data.to_html(classes=['u-full-width'], index=False)
                    .replace('border="1"','border="0"')
                 )
    data_json = data.to_json(orient="records")
    return render_template('awesome.html', data_json=data_json, data_table=table_html)

# Get threshold value and return precision/recall (and data points with pumps predicted to need repairs)
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the threshold from the json, calc precision and recall and
    send it with a response
    """
    data = flask.request.json
    thresh = data["thresh"]

    if Dict_obj.stats_dict.has_key(thresh[0]):
        print "Found previously"
    else:
        print "not found -- calculating " + str(thresh[0])
        (precision,recall) = calculate_stats(thresh[0])
        Dict_obj.stats_dict[thresh[0]] = (precision,recall)
        # Fill precision/recall tables for faster loading in future 
        for thresh_val in np.linspace(0,1.0,101):
            if not Dict_obj.stats_dict.has_key(thresh_val):
                print "calculating for " + str(thresh_val)
                (precision,recall) = calculate_stats(thresh_val)
                Dict_obj.stats_dict[thresh_val] = (precision,recall)
    (precision,recall) = Dict_obj.stats_dict[thresh[0]]
    print("Thresh: " + str(thresh) + " P: " + str(precision) + " R: " + str(recall))

    # Put the result in a nice dict so we can send it as json
    results = {"precision": np.around(precision*100),"recall": np.around(recall*100)}
    return flask.jsonify(results)

def calculate_stats(thresh):
    pred_func = data['func'].values
    prediction = [1 if x>thresh else 0 for x in pred_func]
    not_prediction = [1 if x==0 else 0 for x in prediction]
    actual = [1 if x==2 else 0 for x in data['status'].values]
    not_actual = [1 if x==0 else 0 for x in actual]
    
    
    # TP = broken pumps predicted as broken
    TP = np.multiply(not_actual, not_prediction).sum()
    
    # FP = good pumps predicted as broken
    FP = np.multiply(actual, not_prediction).sum()
    
    # FN = broken pumps predicted as good
    FN = np.multiply(not_actual, prediction).sum()
    
    # TN = good pumps predicted as good
    TN = np.multiply(prediction, actual).sum()

    # precision is % of pumps we predicted as bad that are
    precision = TP * 1. / (TP + FP)
    # recall is % of all the bad pumps we predicted
    recall = TP * 1. / (TP + FN)

    print("TP: " + str(TP) + " FP: " + str(FP) + " FN: " + str(FN) + " TN: " + str(TN))
    return precision,recall

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
#app.debug = True
app.run(host='0.0.0.0', port=5000)
