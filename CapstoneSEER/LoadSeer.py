#SEER database
# SEER data should be loaded into the Data sub-directory of this project. Uses SQlite3
#
#  .\Data
#    \incidence
#       read.seer.research.nov14.sas       <- Data Dictionary
#       *.txt                              <- Data files in fixed width text format
#    \populations
#
# regex to read data dictionary
# \s+@\s+([0-9]+)\s+([A-Z0-9_]*)\s+[$a-z]+([0-9]+)\.\s+/\* (.+?(?= \*/))

import re
import time
import os
import sqlite3
import glob
import pandas as pd
from pandas.io import sql
from MasterSeer import MasterSeer
 

class LoadSeerData(MasterSeer):

    def __init__(self, path=r'.\data', reload=True, testMode=False, verbose=True, batch=10000):

        # user supplied parameters
        self.reload = reload        # deletes and recreates db before start of loading data.
        self.testMode = testMode    # import one file, 100 records and return
        self.verbose = verbose      # prints status messages
        self.batchSize = batch      # number of rows to commit to db in one transation

        if type(path) != str:
            raise TypeError('path must be a string')

        if path[-1] != '\\':
            path += '\\'            # if path does not end with a backslash, add one

        self.path = path

        # open connection to the database
        super().__init__(path, reload, testMode, verbose, batch)
        self.db_conn, self.db_cur = super().init_database(self.reload)

        # TODO
        #if !self.db_conn or !self.db_cur:
            

    # supports specific file or wildcard filename to import all data in one
    # call.
    # path specified is off of the path sent in the constructor so actual
    # filename will be self.path + fname
    def load_data(self, fname=r'incidence\yr1973_2012.seer9\breast.txt'):
        try:
            self.dfDataDict = super().load_data_dictionary()
        except Exception as e:
            print('ERROR loading data dictionary.')
            raise(e)

        if len(self.dfDataDict) == 0:
            raise('Bad Data Dictionary Data')

        timeStart = time.perf_counter()

        totRows = 0
        for fileName in glob.glob(self.path + fname):
            totRows += self.load_one_file(fileName)

        if self.verbose:
            print('Loading Data completed.\n Rows Imported: {0:d} in {1:.1f} seconds.\n Loaded {2:.1f} per sec.'.format(totRows, time.perf_counter() - timeStart, (totRows / (time.perf_counter() - timeStart))))


    def load_one_file(self, fname):
        if self.verbose:
            print('\nStart Loading Data: {0}'.format(fname))

        # Need to get the name of the SEER text file so we can store it into
        # the SOURCE field.
        fileSource = os.path.basename(fname)
        fileSource = os.path.splitext(fileSource)[0]

        try:
            self.db_conn.execute('DROP TABLE {0}'.format(fileSource))
        except:
            pass

        colInfo = []  # hold start, stop byte offset for each field, used by read_fwf
        for off, len in zip(self.dfDataDict.OFFSET, self.dfDataDict.LENGTH):
            colInfo.append((off-1, off-1+len))

        if self.verbose:
            print('Starting read of raw data.')

        dfData = pd.read_fwf(fname, colspecs = colInfo, header=None) #, nrows=100000) #, nrows = self.batchSize, skiprows=totRows)

        # assign column names
        dfData.columns = self.dfDataDict.FIELD_NAME

        if self.verbose:
            print('Starting load of data to database.')

        sql.to_sql(dfData, name=fileSource, con=self.db_conn, index=False, if_exists='append', chunksize=self.batchSize)

        if self.verbose:
            print('\n - Loading completed. Rows Imported: {0:d}'.format(dfData.shape[0]))

        return dfData.shape[0]   #############################


    def create_table(self, tblName):
        # Create the table from the fields read from data dictionary and stored in self.dataDictInfo
        # Make list comma delimited

        fieldList = self.dfDataDict.FIELD_NAME
        delimList = ','.join(map(str, fieldList)) 

        # create the table
        # SECURITY - Not subject to code injection even if Data Dictionary was
        # hacked since create table is the command specified.
        #            Not running any SELECT statements to hack.  Buffer
        #            overflow problems mitigated with checks importing
        #            dictionary.
        self.db_conn.execute('CREATE TABLE {0:s}(SOURCE,'.format(tblName) + delimList + ')')
                            
    def __str__(self, **kwargs):
        pass


if __name__ == '__main__':

    t0 = time.perf_counter()
    seer = LoadSeerData(testMode = False)
    p = seer.load_data(r'incidence\yr1973_2012.seer9\breast.txt')  # load one file
    #p = seer.load_data(r'incidence\yr1973_2012.seer9\*.txt') # load all files

    #seer = LoadSeerData(testMode = False)
    

    print('\nModule Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))