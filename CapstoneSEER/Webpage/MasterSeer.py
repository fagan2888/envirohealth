import re
import time
import os
import sqlite3
import pandas as pd
import numpy as np


class MasterSeer(object):
    ''' Master SEER database class that manages connection and loads raw data into sqlite3 database
    '''

    # database file name on disk
    DB_NAME = 'seer.db'

    def __init__(self, path = r'../data/', reload = True, verbose = True):

        if type(path) != str:
            raise TypeError('path must be a string')

        if path[-1] != '/':
            path += '/'            # if path does not end with a backslash, add one

        self.path = path

        # List to hold lists of [Column Offset, Column Name, Column Length]
        self.dataDictInfo = []
        self.db_conn = None
        self.db_cur = None

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

    def clean_recode_data(self, df, dependent_cutoffs):
        """ clean_recode_data(df)
            params: df - dataframe of seer data to clean
                    dependent_cutoffs - months to use to code SRV_BUCKET
                                        if blank, then use SRV_TIME_MON and don't code into survival buckets.
            returns: cleaned dataframe, and name of new coded dependent variable

            Each cleaning step is on its own line so we can pick and choose what
            steps we want after we decide on variables to study
        """
        # drop all rows that have invalid or missing data
        try:
            df = df.dropna(subset = ['YR_BRTH']) # add column names here as needed
        except Exception as err:
            pass

        try:
            df.LATERAL = df.LATERAL.replace([0, 1,2,3], 1)  # one site = 1
            df.LATERAL = df.LATERAL.replace([4,5,9], 2)     # paired = 2
        except:
            pass

        try:
            df = df[df.O_DTH_CLASS == 0]
        except:
            pass

        try:
            # 0-benign, 1-borderline, 2-in situ, 3-malignant
            df = df[df.BEHANAL != 5]
            df.BEHANAL = df.BEHANAL.replace([3,4,6], 3)
        except:
            pass

        try:
            df = df[df.HST_STGA != 8]
            df = df[df.HST_STGA != 9]
        except:
            pass

        try:
            # 0-negative, 1-borderline,, 2-positive
            df = df[df.ERSTATUS != 4]
            df = df[df.ERSTATUS != 9]
            df.ERSTATUS = df.ERSTATUS.replace(2, 0)
            df.ERSTATUS = df.ERSTATUS.replace(1, 2)
            df.ERSTATUS = df.ERSTATUS.replace(3, 1)
        except:
            pass

        try:
            # 0-negative, 1-borderline,, 2-positive
            df = df[df.PRSTATUS != 4]
            df = df[df.PRSTATUS != 9]
            df.PRSTATUS = df.PRSTATUS.replace(2, 0)
            df.PRSTATUS = df.PRSTATUS.replace(1, 2)
            df.PRSTATUS = df.PRSTATUS.replace(3, 1)
        except:
            pass

        try:
            df.RADIATN = df.RADIATN.replace(7, 0)
            df.RADIATN = df.RADIATN.replace([2,3,4,5], 1)
            df = df[df.RADIATN < 7]
        except Exception as err:
            pass

        try:
            # code as 1 or 2-more than one
            df.NUMPRIMS = df.NUMPRIMS.replace([x for x in range(2,37)], 2)
        except Exception as err:
            pass

        #BG - race recode
        try:
            df.RACE = df.RACE.replace(1,101)
            df.RACE = df.RACE.replace(2,102)
            df.RACE = df.RACE.replace(3,103)
            df.RACE = df.RACE.replace(4,104)
            df.RACE = df.RACE.replace(5,105)
            #df.RACE = df.RACE.replace([[x for x in range(20,32)],6,7,97],107)
            df.RACE = df.RACE.replace([6,7,20,21,22,23,24,25,26,27,28,29,30,31,32,97],107)
            #df.RACE = df.RACE.replace([[x for x in range(8,17)],96],108) doesn't fully work
            df.RACE = df.RACE.replace([8,9,10,11,12,13,14,15,16,17,96],108)
            df.RACE = df.RACE.replace(98,99)
        except:
            pass

        try:
            df.loc[(df['RACE'] == 101) & (df['ORIGIN'] != 0), 'RACE'] = 109
        except:
            pass

        # try:
        #     df[df.RACE == 1][df.ORIGIN != 0] = 109
        # except:
        #     pass

        # if df.RACE == 1 and df.ORIGIN != 0:
        #     df.RACE = df.RACE.replace(1,109)

        # try:
        #     df.RACE = df.RACE.replace({1:101, 2:102, 3:103, 4:104, 5:105, [range(20,32),6,7,97]:107, [range(8,17),96]:108})
        # except:
        #     pass

        #try:
        #    df = df[df.AGE_DX != 999]
        #except:
        #    pass
        #try:
        #    df = df[df.SEQ_NUM != 88]
        #except:
        #    pass
        #try:
        #    df = df[df.GRADE != 9]
        #except:
        #    pass
        #try:
        #    df = df[df.EOD10_SZ != 999]
        #except:
        #    pass
        #try:
        #    df = df[df.EOD10_PN < 95]
        #except:
        #    pass

        #try:
        #    # remove unknown or not performed. reorder 0-neg, 1-borderline, 2-pos
        #    df = df[df.TUMOR_1V in [1,2,3]]
        #    df.TUMOR_1V = df.TUMOR_1V.replace(2, 0)
        #    df.TUMOR_1V = df.TUMOR_1V.replace(1, 2)
        #    df.TUMOR_1V = df.TUMOR_1V.replace(3, 1)
        #except:
        #    pass

        #try:
        #    df.TUMOR_2V = df.TUMOR_2V.replace(7, 0)
        #    df = df[df.RADIATN < 7]
        #except Exception as err:
        #    pass


        # create new dependent column called SRV_BUCKET to hold the survival time value
        # based on the values sent into this function in the dependent_cutoffs list
        # first bucket is set to 0, next 1, etc...
        # Example dependent_cutoffs=[60,120,500]
        #   if survival is less than 60 SRV_BUCKET is set to 0
        #   if survival is >=60 and < 120 SRV_BUCKET is set to 1

        if len(dependent_cutoffs) > 0:
            # create new column of all NaN
            df['SRV_BUCKET'] = np.NaN
            # fill buckets
            last_cut = 0
            for x, cut in enumerate(dependent_cutoffs):
                df.loc[(df.SRV_TIME_MON >= last_cut) & (df.SRV_TIME_MON < cut), 'SRV_BUCKET'] = x
                last_cut = cut
            # assign all values larger than last cutoff to next bucket number
            df['SRV_BUCKET'].fillna(len(dependent_cutoffs), inplace=True)

            dep_col = 'SRV_BUCKET'
            df = df.drop('SRV_TIME_MON', 1)
        else:
            dep_col = 'SRV_TIME_MON'

        # categorical columns to one hot encode, check to make sure they are in df
        #cat_cols_to_encode = list(set(['RACE', 'ORIGIN', 'SEX', 'TUMOR_2V', 'HISTREC']) & set(df.columns))
        #df = self.one_hot_data(df, cat_cols_to_encode)

        df['CENSORED'] = df.STAT_REC == 4
        df = df.drop('STAT_REC', 1)


        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        exc = pd.ExcelWriter('clean.xlsx')
        df.to_excel(exc)
        exc.save()

        return df, dep_col

    def one_hot_data(self, data, cols):
        """ Takes a dataframe and a list of columns that need to be encoded.
            Returns a new dataframe with the one hot encoded vectorized data

            See the following for explanation:
                http://stackoverflow.com/questions/17469835/one-hot-encoding-for-machine-learning
            """
        # check to only encode columns that are in the data
        col_to_process = [c for c in cols if c in data]
        return pd.get_dummies(data, columns = col_to_process,  prefix = col_to_process)
