from flask import Flask, render_template, g, json, request
from ProjectSeer1 import ProjectSeer1
import sqlite3

app = Flask(__name__)

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

@app.route("/showResults")
def showResults():
    return ProjectSeer1(MasterSeer)
    return render_template("results.html")

@app.route("/showResultsTEST", methods=['POST'])
def showResultsTEST():
    _age = request.form['age']
    _radtn = request.form['radtn']
    _race = request.form['race']
    _laterality = request.form['laterality']
    _tumorbehavior = request.form['tumorbehavior']
    _tumorstage = request.form['tumorstage']
    _numprims = request.form['numberprims']
    _erstatus = request.form['erstatus']
    _prstatus = request.form['prstatus']
    # print(_age, _radtn, _race, _laterality, _tumorbehavior, _tumorstage, _numprims, _erstatus, _prstatus)
    user_vars_list = [_age, _radtn, _race, _laterality, _tumorbehavior, _tumorstage, _numprims, _erstatus, _prstatus]
    user_vars_dict = {'AGE_DX': _age, 'RADIATN' : _radtn, 'RACE' : _race, 'LATERAL' : _laterality, 'BEHANAL' : _tumorbehavior, 'HST_STGA' : _tumorstage, 'NUMPRIMS' : _numprims, 'ERSTATUS' : _erstatus, 'PRSTATUS' : _prstatus}
    for var in user_vars_list:
        print(var)
    for key,value in user_vars_dict.items():
        print(key, value)
    return render_template("results.html")

# @app.route("/showResults")
# def showResults():
#     return render_template("results.html")

# @app.route("/showResults",methods=['GET'])
# def calculate():
#     return render_template('results.html')
#     _age = request.form['age']
#     _radtn = request.form['radtn']
#     _race = request.form['race']
#     _laterality = request.form['laterality']
#     _tumorbehavior = request.form['tumorbehavior']
#     _tumorstage = request.form['tumorstage']
#     _numprims = request.form['numberprims']
#     _erstatus = request.form['erstatus']
#     _prstatus = request.form['prstatus']
#
#     fullfields = _age and _radtn and _race and _laterality and _tumorbehavior and _tumorstage and _numprims and _erstatus and _prstatus

    # if fullfields:
    #     return json.dumps({'html':'<span>OK</span>'})
    # else:
    #     return json.dumps({'html':'<span>Err</span>'})

# Run http://localhost:5000/#

# """
# Database code
# """
# DATABASE = '../data/seer.db'
#
# def get_db():
#     db = getattr(g, '_database', None)
#     if db is None:
#         db = g._database = connect_to_database()
#     return db
#
# @app.teardown_appcontext
# def close_connection(exception):
#     db = getattr(g,'_database', None)
#     if db is not None:
#         db.close()

# @app.route('TestQuery')

if __name__ == "__main__":
    app.run()
