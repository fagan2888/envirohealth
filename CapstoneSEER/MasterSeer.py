import re
import time
import os
import sqlite3
import glob
import pandas as pd


class MasterSeer(object):
    """description of class"""

    # database file name on disk
    DB_NAME = 'seer.db'

    def __init__(self, path = r'.\data', reload = True, testMode = False, verbose = True, batch = 5000):
        self.path = path

        # List to hold lists of [Column Offset, Column Name, Column Length]
        self.dataDictInfo = []
        self.db_conn = None
        self.db_cur = None

    def init_database(self, reload):
        try:
            if reload:
                os.remove(self.path + self.DB_NAME)
        except:
            pass

        try:
            #initialize database
            self.db_conn = sqlite3.connect(self.path + self.DB_NAME)
            self.db_cur = self.db_conn.cursor()

            if self.verbose:
                print('Database initialized')

            return self.db_conn, self.db_cur

        except Exception as e:
            print('ERROR connecting to the database: ')
            return None, None


    def load_data_dictionary(self, fname = r'SeerDataDict.txt'):
        
        REGEX_DD = '\$char([0-9]+).'

        t0 = time.perf_counter()

        if self.verbose:
            print('\nStart Load of Data Dictionary')

        # read our custom tab delimited data dictionary
        df = pd.read_csv(self.path + fname, delimiter='\t')

        # drop all rows where IMPORT_0_1 is a zero. 
        df = df[df.IMPORT_0_1 > 0]

        # pre-compile regex to improve performance in loop
        reCompiled = re.compile(REGEX_DD)
        flen = []       # list to hold parsed field lengths

        # add length column
        for row in df.TYPE: 
            fields = reCompiled.match(row)
            if fields:
                x = int(fields.groups()[0])
                flen.append(x)

        # check to make sure we read the correct amount of field lengths
        if len(flen) != len(df):
            print('ERROR reading field lengths')
            return None

        # add length column to dataframe
        df['LENGTH'] = flen

        if self.verbose:
            print('Data Dictionary loaded in {0:5.4f} sec.'.format(time.perf_counter() - t0), flush=True)

        return df
