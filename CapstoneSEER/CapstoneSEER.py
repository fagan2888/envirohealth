#SEER database
# SEER data should be loaded into the Data sub-directory of this project.
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

class LoadSeerData:

    def __init__(self, path = r'.\data', reload = True, testMode = False, verbose = True):
        if type(path) != str:
            raise TypeError('path must be a string')

        if path[-1] != '\\':
            path += '\\'

        self.path = path

        # used to read in data dictionary, used to parse actual data files.
        self.SeerDataDictRegexPat = '\s+@\s+([0-9]+)\s+([A-Z0-9_]*)\s+[$a-z]+([0-9]+)\.\s+/\* (.+?(?=\*/))'
        self.dataDictFieldNames = ['Offest', 'ColName', 'Length']
        self.colOffset = []
        self.colName = []
        self.colLength = []

        self.testMode = testMode
        self.verbose = verbose

        # open connection to the database
        self.init_database(reload)


    def init_database(self, reload):
        try:
            if reload:
                os.remove(self.path + 'seer.db')

            #initialize database
            self.db_conn = sqlite3.connect(self.path+'seer.db')
            self.db_cur = self.db_conn.cursor()

            if self.verbose:
                print('Database initialized\n')
        except Exception as e:
            print('ERROR connecting to the database: ' + e.strerror)
            raise(e)



    def load_data_dictionary(self, fname = r'incidence\read.seer.research.nov14.sas'):
        if self.verbose:
            print('Start Load of Data Dictionary\n')

        # TODO look into a better way to read this file, don't like the if elif structure
        with open(self.path + fname) as fDataDict:
            for line in fDataDict:
                fields = re.match(self.SeerDataDictRegexPat, line)
                if fields:
                    for x in range(4):
                        if x == 0:
                            self.colOffset.append(int(fields.groups()[x])-1)  # change to 0 offset, Data Dict starts at 1
                        elif x == 1:
                            self.colName.append(fields.groups()[x])
                        elif x == 2:
                            self.colLength.append(int(fields.groups()[x]))

        if self.verbose:
            print('Data Dictionary loaded\n')


    # supports specific file or wildcard filename to import all data in one call.
    # path specified is off of the path sent in the constructor so actual filename will be self.path + fname
    def load_data(self, fname = r'incidence\yr1973_2012.seer9\breast.txt'):
        try:
            self.load_data_dictionary()
        except Exception as e:
            print('ERROR loading data dictionary: ' + e.strerror)
            raise(e)

        if not (len(self.colOffset) == len(self.colLength) == len(self.colName)) and len(self.colName) > 0:
            raise('Bad Data Dictionary Data')

        # create the table in the db
        self.create_table()

        totRows = 0
        for fileName in glob.glob(self.path + fname):
            totRows += self.load_one_file(fileName)

        if self.verbose:
            print('\nLoading Data completed. Rows Imported: {}\n'.format(totRows))


    def load_one_file(self, fname):
        if self.verbose:
            print('Start Loading Data: {}\n'.format(fname))

        # Need to get the name of the SEER text file so we can store it into the SOURCE field.
        fileSource = os.path.basename(fname)
        fileSource = '\'' + os.path.splitext(fileSource)[0] + '\''
        
        # pre-build the sql statement outside of the loop so it is only called once
        #   get list of field names for INSERT statement
        fieldList = ','.join(map(str, self.colName))

        command = 'INSERT INTO seer(SOURCE,' + fieldList + ') values (' + '?,' * len(self.colName) + '?)'

        # create variables needed in loop
        batchSize = 1000             # INSERT 1000 at a time.
        rowValues = []               # hold one records values
        multipleRowValues = []       # hold batchSize lists of rowValues to commit to DB in one transaction
        totRows = 0                   

        # open SEER fixed width text file
        with open(fname, 'r') as fData:
            
            for line in fData:
                totRows += 1
                rowValues.clear()
                rowValues.append(fileSource)  # first field is the SEER data file name i.e. breast or respir

                # iterate through all of the fields in the text file and store to rowValues list
                for fldNum in range(len(self.colOffset)):
                    field = line[self.colOffset[fldNum]:self.colOffset[fldNum]+self.colLength[fldNum]]
                    rowValues.append('\'' + field + '\'')

                # store this one row list of values to the list of lists for batch insert
                multipleRowValues.append(rowValues)

                # commit to DB in batchSize batches to speed performance
                if totRows % batchSize == 0:
                    self.db_cur.executemany(command, multipleRowValues)
                    self.db_conn.commit()
                    multipleRowValues.clear()
                    if self.verbose:
                        print('', end='.', flush=True)

                # if in testMode, exit loop after 100 records are stored
                if totRows > 100 and self.testMode:
                    self.db_cur.executemany(command, multipleRowValues)
                    self.db_conn.commit()
                    break

        if self.verbose:
            print('\nLoading Data file completed. Rows Imported: {}\n'.format(totRows))

        return totRows


    def create_table(self):
        # Create the table from the fields read from data dictionary and stored in self.colName
        # Make colName list comma delimited
        delimList = ','.join(map(str, self.colName)) 

        # create the table
        self.db_conn.execute('create table seer(SOURCE,' + delimList + ')')
                            
    def __str__(self, **kwargs):
        pass


def test():
    seer = LoadSeerData()
    #p = seer.load_data(r'incidence\yr1973_2012.seer9\breast.txt')  # load one file
    p = seer.load_data(r'incidence\yr1973_2012.seer9\*.txt')   # load all files


if __name__ == '__main__':

    timeStart = time.clock();
    test()
    print('Elapsed Time: ', time.clock() - timeStart)