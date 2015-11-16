import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MasterSeer import MasterSeer
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.linear_model import LinearRegression


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






if __name__ == '__main__':

    t0 = time.perf_counter()

    seer = ExploreSeer()
    seer.describe_data()

    del seer
    print('\nExploreSeer Module Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))
