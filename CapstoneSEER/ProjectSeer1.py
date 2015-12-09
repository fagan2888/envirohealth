from MasterSeer import MasterSeer
import numpy as np
import matplotlib.pyplot as plt
import patsy as pt
from lifelines import AalenAdditiveFitter

class ProjectSeer1(MasterSeer):

    def __init__(self, path=r'./data/', verbose=True, sample_size = 5000):
        # user supplied parameters
        self.verbose = verbose          # prints status messages

        # open connection to the database
        super().__init__(path, False, verbose=verbose)
        self.db_conn, self.db_cur = super().init_database(False)

        self.model = None
        self.sample_size = sample_size


    def __del__(self):
        super().__del__()

    def load_and_clean_data(self, censored=True):

        where = 'SRV_TIME_MON < 1000 AND HST_STGA < 8 AND DTH_CLASS < 9 AND ERSTATUS < 4 AND PRSTATUS < 4'
        if not censored:
            where += ' AND DATE_yr < 2008'

        df = super().load_data(col  = ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS', 
                                       'SRV_TIME_MON', 'STAT_REC'], 
                               cond = where, sample_size = self.sample_size)

        df, dependent = super().clean_recode_data(df, [])

        return df, dependent

    def run_survival_curve(self, df):
        ''' used for testing only'''

        aaf = AalenAdditiveFitter()

        modelspec = 'YR_BRTH + AGE_DX + RADIATN + HISTREC + ERSTATUS + PRSTATUS + BEHANAL + HST_STGA + NUMPRIMS'
        X = pt.dmatrix(modelspec, df, return_type='dataframe')
        X = X.join(df[['SRV_TIME_MON','CENSORED']])
        aaf.fit(X, 'SRV_TIME_MON', 'CENSORED')

        # INSERT VALUES TO TEST HERE                
        test = np.array([[ 1., 1961., 52., 0, 0., 2., 1., 0., 4., 2.]])

        aaf.predict_survival_function(test).plot();
        plt.show()

        exp = aaf.predict_expectation(test)
        print(exp)

        return


    def prepare_model(self):
        # get the data and clean it
        df, dep = self.load_and_clean_data(self)

        # create the model
        aaf = AalenAdditiveFitter()

        # define fields for the model
        modelspec = 'YR_BRTH + AGE_DX + RADIATN + HISTREC + ERSTATUS + PRSTATUS + BEHANAL + HST_STGA + NUMPRIMS'
        X = pt.dmatrix(modelspec, df, return_type='dataframe')
        X = X.join(df[['SRV_TIME_MON','CENSORED']])

        # fit the model
        print('Creating Aalen Additive Model')
        aaf.fit(X, 'SRV_TIME_MON', 'CENSORED')

        return aaf


    def process_patient(self, pat_data):
        ''' process_patient(pat_data)
               fits the model if not already done, estimates survival of individual patient
               and displays patient's survival analysis graph.

            params: pat_data-numpy array of patient specific values for the following columns
                        ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS',
                         'PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS']
            returns: nothing
        '''
        if not self.model:
            self.model = self.prepare_model()

        cols = ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS']
        for i, col in enumerate(cols):
            print('{0:10}: {1:.0f}'.format(col, pat_data[0][i+1]))

        exp = self.model.predict_expectation(test)
        print('Expected survival: {0:.1f} months'.format(exp[0][0]))

        self.model.predict_survival_function(test).plot();
        plt.xlabel('Months')
        plt.show()


if __name__ == '__main__':

    def_rows = 10000

    try:
        rows = int(input('Enter Sample Size [{0}]: '.format(def_rows)))
        if rows < 100 or rows > 1000000:
            rows = def_rows
    except:
        rows = def_rows
    
    seer = ProjectSeer1(sample_size = rows)

    #run first person
    test = np.array([[ 1., 1961., 52., 0, 0., 2., 1., 0., 4., 2.]])
    seer.process_patient(test)

    # run second person
    test = np.array([[ 1., 1941., 72., 0, 0., 2., 1., 0., 3., 2.]])
    seer.process_patient(test)
    
    del seer