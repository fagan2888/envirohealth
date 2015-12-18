from MasterSeer import MasterSeer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy as pt
import os
import random
from lifelines import AalenAdditiveFitter #, CoxPHFitter
#from lifelines.utils import k_fold_cross_validation

class ProjectSeer1(MasterSeer):

    def __init__(self, path=r'../data/', verbose=True, sample_size = 5000):
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

        where = 'SRV_TIME_MON < 1000 AND HST_STGA < 8 AND O_DTH_CLASS = 0 AND ERSTATUS < 4 AND PRSTATUS < 4'
        if not censored:
            where += ' AND DATE_yr < 2008'

        df = super().load_data(col  = ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS', 'RACE', 'ORIGIN',
                                       'SRV_TIME_MON', 'STAT_REC'],
                               cond = where, sample_size = self.sample_size)

        df, dependent = super().clean_recode_data(df, [])

        return df, dependent

    def run_survival_curve(self, df):
        ''' used for testing only'''

        aaf = AalenAdditiveFitter()

        modelspec = 'YR_BRTH + AGE_DX + RADIATN + HISTREC + ERSTATUS + PRSTATUS + BEHANAL + HST_STGA + NUMPRIMS + RACE'
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

    def score_model(self):
        # get the data and clean it
        temp = self.sample_size
        self.sample_size = 100000
        df, dep = self.load_and_clean_data()
        self.sample_size = temp

        # create the model
        aaf = AalenAdditiveFitter()
        cph = CoxPHFitter()

        # define fields for the model
        modelspec = 'YR_BRTH + AGE_DX + RADIATN + HISTREC + ERSTATUS + PRSTATUS + BEHANAL + HST_STGA + NUMPRIMS + RACE'
        X = pt.dmatrix(modelspec, df, return_type='dataframe')
        X = X.join(df[['SRV_TIME_MON','CENSORED']])

        scores = k_fold_cross_validation(aaf, X, 'SRV_TIME_MON', event_col='CENSORED', k=5)
        print('\nCross Validation Scores: ')
        print(scores)
        print('Score Mean: {0:.4}'.format(np.mean(scores)))
        print('Score SD  : {0:.4}'.format(np.std(scores)))

        return


    def prepare_model(self):

        # get the data and clean it
        df, dep = self.load_and_clean_data()

        # create the model
        aaf = AalenAdditiveFitter()

        # define fields for the model
        modelspec = 'YR_BRTH + AGE_DX + RADIATN + HISTREC + ERSTATUS + PRSTATUS + BEHANAL + HST_STGA + NUMPRIMS + RACE'
        X = pt.dmatrix(modelspec, df, return_type='dataframe')
        X = X.join(df[['SRV_TIME_MON','CENSORED']])

        # fit the model
        if self.verbose:
            print('Creating Aalen Additive Model')

        aaf.fit(X, 'SRV_TIME_MON', 'CENSORED')

        return aaf


    def process_patient(self, pat_data, dyn_img):
        ''' process_patient(pat_data)
               fits the model if not already done, estimates survival of individual patient
               and displays patient's survival analysis graph.

            params: pat_data-numpy array of patient specific values for the following columns
                        ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS',
                         'PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS', 'RACE']
            returns: expected survival time in months
        '''
        try:
            os.remove('./static/plot.png')
        except:
            pass

        if not self.model:
            self.model = self.prepare_model()

        exp = self.model.predict_expectation(pat_data)

        if self.verbose:
            cols = ['YR_BRTH','AGE_DX','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS','RACE']
            if self.verbose:
                for i, col in enumerate(cols):
                    print('{0:10}: {1:.0f}'.format(col, pat_data[0][i+1]))

            exp_srv_mnth = ('Expected survival: {0:.1f} months'.format(exp[0][0]))

            print(exp_srv_mnth)

            self.model.predict_survival_function(pat_data).plot(legend=None, color="#3F5D7D");
            plt.xlabel('Months')
            plt.ylabel('Survival Percentage')
            plt.title('Survival Analysis')

            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            lbl = 'Expected: {0:.1f}'.format(exp[0][0])

            # comment out the following line to revoce the vertical line as estimated survival
            ax.axvline(x=exp[0][0],linewidth=3, color='b', ymin=0.15, ymax=0.65, label=lbl, dashes=(1,3)) #dashes='--')#, label='Expected: ' + str(exp[0][0]))

            # plt.savefig('./static/' + nameappend + '.png', bbox_inches="tight")
            plt.savefig(dyn_img, bbox_inches="tight")
            # if self.verbose:
            #     plt.show()

        return exp[0][0]


if __name__ == '__main__':

    #def_rows = 10000

    #try:
    #    rows = int(input('Enter Sample Size [{0}]: '.format(def_rows)))
    #    if rows < 100 or rows > 1000000:
    #        rows = def_rows
    #except:
    #    rows = def_rows

    seer = ProjectSeer1(sample_size = 10000, verbose=True)

    #seer.score_model()

    ##run first person
    test = np.array([[ 1., 1961., 54., 0, 0., 2., 1., 0., 4., 2., 101.]])
    seer.process_patient(test)

    ## run second person
    #test = np.array([[ 1., 1961., 54., 0, 0., 2., 1., 0., 1., 2., 101.]])
    #seer.process_patient(test)

    del seer
