import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MasterSeer import MasterSeer
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.linear_model import LinearRegression
from lifelines.plotting import plot_lifetimes
from lifelines import KaplanMeierFitter
from numpy.random import uniform, exponential

class ExploreSeer(MasterSeer):

    def __init__(self, path=r'./data/', testMode=False, verbose=True, sample_size=5000):

        # user supplied parameters
        self.testMode = testMode        # import one file, 500 records and return
        self.verbose = verbose          # prints status messages
        self.sample_size = sample_size  # number of rows to pull for testing

        if type(path) != str:
            raise TypeError('path must be a string')

        if path[-1] != '/':
            path += '/'            # if path does not end with a backslash, add one

        self.path = path

        # open connection to the database
        super().__init__(path, False, verbose=verbose)
        self.db_conn, self.db_cur = super().init_database(False)


    def __del__(self):
        super().__del__()


    def describe_data(self, source = 'breast'):
        """ describe_data(source)
            params: source - table name in seer database, defaults to 'breast'
            returns: panda.DataFrame.describe() data and the dataframe

            Called from prepare_test_train_sets()
            the describe data is stored in an excel file. 
        """
        xls_name = source + '_seer_describe.xlsx'

        df = super().load_data(source)
        desc = df.describe(include='all')
        exc = pd.ExcelWriter(xls_name)
        desc.to_excel(exc)
        exc.save()
        print("Data description saved to {0}".format(xls_name))

        return desc, df


    def find_features(self, X, y, plot = True):
        """
            work in progress - not completed
        """
        X_indices = np.arange(X.shape[-1])

        col = list(X.columns.values)
        test = 2

        if test == 1:
            selector = SelectPercentile(f_classif, percentile=10)
            selector.fit(np.array(X), y)
            values = np.nan_to_num(selector.pvalues_)
        else:
            model = LinearRegression()
            model.fit(np.array(X), y)
            selector = SelectFromModel(f_classif)
            selector.fit(np.array(X), y)

        values = np.nan_to_num(selector.pvalues_)

        if plot:
            #scores = -np.log10(values)
            #scores /= scores.max()

            fig, ax = plt.subplots()
            #ax.set_xticks(col)
            ax.set_xticklabels(col, rotation='vertical')
            ax.set_title(r'Univariate score')

            ax.bar(X_indices - .45, values, width=.2, color='g')
            plt.show()

        for i, val in enumerate(values):
            print("{0}  {1:.2f}".format(col[i], val))

        return

    def plot_survival(self):

        df = super().load_data(col  = ['YR_BRTH','AGE_DX','LATERAL','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS', 'SRV_TIME_MON', 'SRV_TIME_MON_PA', 'DTH_CLASS', 'O_DTH_CLASS', 'STAT_REC'], 
                               cond = 'SRV_TIME_MON < 1000 AND HST_STGA < 8 AND DTH_CLASS < 9 AND ERSTATUS < 4 AND PRSTATUS < 4', sample_size = 100000)

        kmf = KaplanMeierFitter()

        try:
            df.RADIATN = df.RADIATN.replace(7, 0)
            df = df[df.RADIATN < 7] 
        except Exception as err:
            pass

        # 0-negative, 1-borderline,, 2-positive
        df = df[df.ERSTATUS != 4]
        df = df[df.ERSTATUS != 9]
        df.ERSTATUS = df.ERSTATUS.replace(2, 0)
        df.ERSTATUS = df.ERSTATUS.replace(1, 2)
        df.ERSTATUS = df.ERSTATUS.replace(3, 1)

        # 0-negative, 1-borderline,, 2-positive
        df = df[df.PRSTATUS != 4]
        df = df[df.PRSTATUS != 9]
        df.PRSTATUS = df.PRSTATUS.replace(2, 0)
        df.PRSTATUS = df.PRSTATUS.replace(1, 2)
        df.PRSTATUS = df.PRSTATUS.replace(3, 1)

        rad = df.RADIATN > 0
        er  = df.ERSTATUS > 0
        pr  = df.PRSTATUS > 0

        st0  = df.HST_STGA == 0
        st1  = df.HST_STGA == 1
        st2  = df.HST_STGA == 2
        st4  = df.HST_STGA == 4

        age = df.AGE_DX < 50

        #print(df.head())
        #print(rad.head())
        #print(er.head())
        #print(st.head())

        df['SRV_TIME_YR'] = df['SRV_TIME_MON'] / 12
        T = df['SRV_TIME_YR']
        #C = (np.logical_or(df.DTH_CLASS == 1, df.O_DTH_CLASS == 1))
        C = df.STAT_REC == 4

        #print(T.head(20))
        #print(C.head(20))
        #print(df.DTH_CLASS.head(20))
        #print(df.O_DTH_CLASS.head(20))
        #print(df.describe())

         
        f, ax = plt.subplots(5, sharex=True, sharey=True)
        ax[0].set_title("Lifespans of cancer patients");

        # radiation
        kmf.fit(T[rad], event_observed=C[rad], label="Radiation")
        kmf.plot(ax=ax[0]) #, ci_force_lines=True)
        kmf.fit(T[~rad], event_observed=C[~rad], label="No Radiation")
        kmf.plot(ax=ax[0]) #, ci_force_lines=True)

        # ER Status
        kmf.fit(T[er], event_observed=C[er], label="ER Positive")
        kmf.plot(ax=ax[1]) #, ci_force_lines=True)
        kmf.fit(T[~er], event_observed=C[~er], label="ER Negative")
        kmf.plot(ax=ax[1]) #, ci_force_lines=True)

        # PR Status
        kmf.fit(T[pr], event_observed=C[pr], label="PR Positive")
        kmf.plot(ax=ax[2]) #, ci_force_lines=True)
        kmf.fit(T[~pr], event_observed=C[~pr], label="PR Negative")
        kmf.plot(ax=ax[2]) #, ci_force_lines=True)

        # stage
        kmf.fit(T[st0], event_observed=C[st0], label="Stage 0")
        kmf.plot(ax=ax[3]) #, ci_force_lines=True)
        kmf.fit(T[st1], event_observed=C[st1], label="Stage 1")
        kmf.plot(ax=ax[3]) #, ci_force_lines=True)
        kmf.fit(T[st2], event_observed=C[st2], label="Stage 2")
        kmf.plot(ax=ax[3]) #, ci_force_lines=True)
        kmf.fit(T[st4], event_observed=C[st4], label="Stage 4")
        kmf.plot(ax=ax[3]) #, ci_force_lines=True)

        # age
        kmf.fit(T[age], event_observed=C[age], label="Age < 50")
        kmf.plot(ax=ax[4]) #, ci_force_lines=True)
        kmf.fit(T[~age], event_observed=C[~age], label="Age >= 50")
        kmf.plot(ax=ax[4]) #, ci_force_lines=True)

        ax[0].legend(loc=3,prop={'size':10})
        ax[1].legend(loc=3,prop={'size':10})
        ax[2].legend(loc=3,prop={'size':10})
        ax[3].legend(loc=3,prop={'size':10})
        ax[4].legend(loc=3,prop={'size':10})

        ax[len(ax)-1].set_xlabel('Survival in years')

        f.text(0.04, 0.5, 'Survival %', va='center', rotation='vertical')
        plt.tight_layout()

        plt.ylim(0,1);
        plt.show()

        f, ax = plt.subplots(2, sharex=True, sharey=True)

        df.hist('SRV_TIME_YR', by=df.STAT_REC != 4, ax=(ax[0], ax[1]))
        ax[0].set_title('Histogram of Non Censored Patients')
        ax[0].set_ylabel('Number of Patients')

        ax[1].set_ylabel('Number of Patients')
        ax[1].set_title('Histogram of Censored Patients')
        ax[1].set_xlabel('Survival in Years')
        plt.show()

        return

        # second plot of survival

        fig, ax = plt.subplots(figsize=(8, 6))

        cen = df[df.STAT_REC != 4].SRV_TIME_MON
        nc = df[df.STAT_REC == 4].SRV_TIME_MON
        cen = cen.sort_values()
        nc = nc.sort_values()

        ax.hlines([x for x in range(len(nc))] , 0, nc , color = 'b', label='Uncensored');
        ax.hlines([x for x in range(len(nc), len(nc)+len(cen))], 0, cen, color = 'r', label='Censored');

        ax.set_xlim(left=0);
        ax.set_xlabel('Months');
        ax.set_ylim(-0.25, len(df) + 0.25);
        ax.legend(loc='best');
        plt.show()

        return



if __name__ == '__main__':

    t0 = time.perf_counter()

    seer = ExploreSeer(sample_size=10000)
    #seer.describe_data()
    seer.plot_survival()

    del seer
    print('\nExploreSeer Module Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))
