from flask import Flask, render_template, g, json, request
import MasterSeer
import sqlite3
from ProjectSeer1 import ProjectSeer1
import numpy as np
import matplotlib.pyplot as plt
import patsy as pt
from lifelines import AalenAdditiveFitter
import random
import os
import pandas as pd

app = Flask(__name__)
seer = None

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/showCalcPage")
def showCalcPage():
    return render_template('CalcPage.html')

@app.route("/showInformation")
def showInformation():
    return render_template('Information.html')

@app.route("/showResults", methods=['POST'])
def showResults():
    _agedx = request.form['agedx']
    _yrbrth = request.form['yrbrth']
    _radtn = request.form['radtn']
    _race = request.form['race']
    _histology = request.form['histology']
    _tumorbehavior = request.form['tumorbehavior']
    _tumorstage = request.form['tumorstage']
    _numprims = request.form['numberprims']
    _erstatus = request.form['erstatus']
    _prstatus = request.form['prstatus']

    dyn_img = str('./static/' + str(random.random()) + '.png')
    usr_var_array = np.array([[1.,_yrbrth,_agedx,_radtn,_histology,_erstatus,_prstatus,_tumorbehavior,_tumorstage,_numprims,_race]], dtype=np.float64)
    # seer = ProjectSeer1(sample_size = 1000, verbose=True)
    res = seer.process_patient(usr_var_array, dyn_img)
    srv_mnth = str(res)
    srv_prd = srv_mnth.index('.') + 3
    red_srv_mnth = srv_mnth[:srv_prd]
    return render_template("results.html", red_srv_mnth = red_srv_mnth, dyn_img = dyn_img)

if __name__ == "__main__":
    seer = ProjectSeer1(sample_size = 10000, verbose=True)
    test = np.array([[ 1., 1961., 54., 0, 0., 2., 1., 0., 4., 2., 101.]])
    x = seer.process_patient(test, 'test.png')
    print (x)
    app.run()
