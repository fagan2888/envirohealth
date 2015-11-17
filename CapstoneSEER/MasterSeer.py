import re
import time
import os
import sqlite3
import pandas as pd


class MasterSeer(object):
    ''' Master SEER database class that manages connection and loads raw data into sqlite3 database
    '''

    # database file name on disk
    DB_NAME = 'seer.db'

    def __init__(self, path = r'./data/', reload = True, verbose = True, batch = 5000):
        self.path = path

        # List to hold lists of [Column Offset, Column Name, Column Length]
        self.dataDictInfo = []
        self.db_conn = None
        self.db_cur = None
        self.batch = batch

    def __del__(self):
        self.db_conn.close()


    def init_database(self, reload):
        ''' creates a database connection and cursor to sqlite3 database.

            params: reload - if True, deletes all data and creates a new empty database
                           - if False, opens existing db with data
            returns: database connection and database cursor
        '''
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
        ''' loads the data dictionary describing the raw SEER flat file data
            params: fname - the name of the tab delimited file containing the column definitions.

            returns: dataframe of data dictionary from tab delimited file
        '''
        
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


    def load_data(self, source='breast', col=[], cond="YR_BRTH > 0", sample_size=5000, all=False):
        ''' loads data from the sqlite seer database
            params: source - name of table to read from. default 'breast'
                    col - list of column names to return in SELECT statement
                    cond - string for WHERE clause of SELECT statement (do not include the keyword WHERE in the string)
                           defaults to 'YR_BRTH > 0' 
                    sample_size - number of records to return
                    all - if set to true, return entire table and ignore sample_size

            returns: dataframe of data
        '''
        if col:
            col = ','.join(map(str, col)) 
        else:
            col = "*"

        if all:
            limit = ""
            randomize = ""
        else:
            limit = "LIMIT " + str(sample_size)
            randomize = "ORDER BY RANDOM()"

        df = pd.read_sql_query("SELECT {0} FROM {1} WHERE {2} {3} {4}".format(col, source, cond, randomize, limit), self.db_conn)

        return df

