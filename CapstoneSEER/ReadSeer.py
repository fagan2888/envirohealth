import time
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from MasterSeer import MasterSeer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BaseDiscreteNB, BernoulliNB
import math
import itertools
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation


class ReadSeer(MasterSeer):

    def __init__(self, path=r'.\data', reload=False, testMode=False, verbose=True, batch=5000, chunkSize=50000):

        # user supplied parameters
        self.reload = reload        # deletes and recreates db before start of loading data.
        self.testMode = testMode    # import one file, 100 records and return
        self.verbose = verbose      # prints status messages
        self.batchSize = batch      # number of rows to commit to db in one transation
        self.batchSize = chunkSize  # number of rows to load into pandas memory at one time

        if type(path) != str:
            raise TypeError('path must be a string')

        if path[-1] != '\\':
            path += '\\'            # if path does not end with a backslash, add one

        self.path = path

        # open connection to the database
        super().__init__(path, False, testMode, verbose, batch)
        self.db_conn, self.db_cur = super().init_database(False)

    def __del__(self):
        self.db_conn.close()


    def describe_data(self, source = 'BREAST'):
        xls_name = source + '_seer_describe.xlsx'

        if self.testMode:
            df = pd.read_sql_query("SELECT * from {0} where yr_brth > 0 ORDER BY RANDOM() LIMIT 1000".format(source), self.db_conn)
        else:
            df = pd.read_sql_query("SELECT * from {0}".format(source), self.db_conn)


        desc = df.describe(include='all')
        exc = pd.ExcelWriter(xls_name)
        desc.to_excel(exc)
        exc.save()
        print("Data description saved to {0}".format(xls_name))

        #print(df.max(0))
        #df.hist('YR_BRTH')
        #plt.show()
        #pd.DataFrame.hist(df) #, 'YR_BRTH')
        #df1 = df.applymap(lambda x: isinstance(x, (int, float))).all(1)
        return desc

    def _test_code(self): ## not working used for testing
        res = []

        x = pd.Series()
        s = ' '

        for col in df.columns.values:
            s = df[col]
            tot = 0
            na = 0

            isalpha = False

            if type(s) is object:
                z = s.str.isnumeric()
                #isalpha = 

            for val in s:
                if val == '':
                    na += 1
                else:
                    if type(val) is not np.int64 and not val.isdigit():
                        isalpha = True
                tot += 1

            res.append([col, na, tot])


        with open('describe.txt', 'w') as f:
            f.write('Variable              Blanks     Total    %Blank\n')
            for r in res:
                pct_blank = (float(r[1])/float(r[2])) * 100.0
                f.write('{0:20s}   {1:5d}   {2:7d}    {3:3.1f}  {4}\n'.format(r[0], r[1], r[2], pct_blank, isalpha))

        #print(res)
        df[1][3] = None
        #df.fillna('None')
        #df.replace([''], [None])
        print(df.head())
        #c = df.count()
        #print(c)
        #x = np.count_nonzero(df.isnull().values)
        #with open('describe.txt', 'w') as f:
        #    f.write(df.describe().__str__())
        #print(df.describe())
        return


    def get_cols(self, desc):
        exclude = ['SRV_TIME_MON', 'CASENUM', 'REG', 'SITEO2V', 'EOD13', 'EOD2','ICDOT10V', 'DATE_mo', 'SRV_TIME_MON_PA', 'SEQ_NUM']
        cols = []
        for field in desc:
            x = desc[field]['count']
            if desc[field]['count'] > desc['CASENUM']['count'] / 2 and \
               desc[field]['25%'] != desc[field]['75%'] and \
               field not in exclude:
                cols.append(field)
        return cols

    def test_bayes(self, cols, source = 'BREAST'):
        # name of excel file to dump results
        xls_name = source + '_seer_bayes.xlsx'
        # variable to predict
        dependent = 'SRV_TIME_MON'
        # number of records to pull from db, later separated into test/train sets
        records = 1000

        # bayes routines to test
        styles = [MultinomialNB, GaussianNB, BernoulliNB]
        style_names = ['MultinomialNB', 'GaussianNB', 'BernoulliNB']

        # pull relevent fields from database using random rows.
        delimList = ','.join(map(str, cols)) 
        df = pd.read_sql_query("SELECT {0}, {1} \
                                FROM {2} \
                                WHERE AGE_DX < 100 \
                                AND EOD10_SZ BETWEEN 1 AND 100 \
                                AND SRV_TIME_MON BETWEEN 1 AND 1000 \
                                ORDER BY RANDOM() \
                                LIMIT {3}".format(delimList, dependent, source, records), self.db_conn)

        #self.find_clfs(df)
        #return

        # split data frame into train and test sets (80/20)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(df, df[dependent].values, test_size=0.2, random_state=0)
        # test 3 features at a time
        num_var = 3

        col_cnt = len(cols)
        # formula for number of combinations: nCr = n! / r! (n - r)! 
        tot_cnt = (math.factorial(col_cnt) / (math.factorial(num_var) * math.factorial(col_cnt - num_var))) * len(styles)
        print("Processing: {0} tests.".format(int(tot_cnt)))

        res = []
        counter = 0
        for style in range(len(styles)):
            style_fnc = styles[style]
            print("Testing: {0}".format(style_names[style]))
            for combo in itertools.combinations(cols, num_var):
                try: 
                    # train this model
                    x = X_train[ [k for k in combo[:num_var]] ].values
                    y = y_train
                    model = style_fnc()
                    model.fit(x,y)
                    # now test and score it
                    x = X_test[ [k for k in combo[:num_var]] ].values
                    y = y_test
                    z = model.score(x, y)
                    res.append([style_names[style], z, [k for k in combo[:num_var]]])
                    counter += 1
                    if counter % 100 == 0:
                        print("Completed: {0}".format(counter, flush=True), end = '\r')
                except Exception as err:
                    counter += 1
                    #print(err)

        # store trial results to excel
        res_df = pd.DataFrame(res)
        exc = pd.ExcelWriter(xls_name)
        res_df.to_excel(exc)
        exc.save()
        print("\nAll Completed: {0}  Results stored in: {1}".format(counter, xls_name))




if __name__ == '__main__':

    t0 = time.perf_counter()

    seer = ReadSeer(testMode = True)

    desc = seer.describe_data()
    cols = seer.get_cols(desc)
    seer.test_bayes(cols)

    print('\nReadSeer Module Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))