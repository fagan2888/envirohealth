from flask import Flask, render_template, g
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
def shoeInformation():
    return render_template("results.html")

# Run http://localhost:5000/#

"""
Database code
"""

DATABASE = 'C:/Users/bretg_000/Programming/CapstoneSEER/envirohealth/CapstoneSEER/data'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = connect_to_database()
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g,'_database', None)
    if db is not None:
        db.close()

# @app.route('TestQuery')



if __name__ == "__main__":
    app.run()
