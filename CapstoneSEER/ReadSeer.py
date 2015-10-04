#import re
import time
#import os
import sqlite3
#import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from MasterSeer import MasterSeer


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


    def describe_data(self, source = 'BREAST'):

        if self.testMode:
            df = pd.read_sql_query("SELECT * from {0} where yr_brth > 0 limit 1000".format(source), self.db_conn)
        else:
            df = pd.read_sql_query("SELECT * from {0}".format(source), self.db_conn)


        desc = df.describe(include='all')

        exc = pd.ExcelWriter('seer_describe.xlsx')

        desc.to_excel(exc)

        exc.save()


        #print(df.max(0))

        #df.hist('YR_BRTH')
        #plt.show()

        #pd.DataFrame.hist(df) #, 'YR_BRTH')


        #df1 = df.applymap(lambda x: isinstance(x, (int, float))).all(1)

        return  ##########################################################


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
           
        return

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




if __name__ == '__main__':

    t0 = time.perf_counter()

    #plt.ion()

    #ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

    #ts = ts.cumsum()




    #t = arange(0.0, 2.0, 0.01)
    #s = sin(2*pi*t)
    #plot(t, s)

    #xlabel('time (s)')
    #ylabel('voltage (mV)')
    #title('About as simple as it gets, folks')
    #grid(True)
    #savefig("test.png")
    #show()



    seer = ReadSeer(testMode = False)

    seer.describe_data()

    print('\nReadSeer Module Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))