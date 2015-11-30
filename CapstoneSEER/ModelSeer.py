import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MasterSeer import MasterSeer
import math
import itertools
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.cross_validation import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support

class ModelSeer(MasterSeer):

    def __init__(self, path=r'./data/', testMode=False, verbose=True, sample_size=5000, where="DATE_yr < 2008"):

        # user supplied parameters
        self.testMode = testMode        # import one file, 500 records and return
        self.verbose = verbose          # prints status messages
        self.sample_size = sample_size  # number of rows to pull for testing
        self.where = where              # filter for SQL load of data

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


    def prepare_test_train_sets(self, source, dependent, test_pct = .20, return_one_df=False, cols = [], dependent_cutoffs=[60]):
        """ prepare_test_train_sets(source, dependent):
            params:  source - table name in seer database, defaults to 'breast'
                     dependent - name of field we are testing for, need to remove from X and assign to Y
                     test_pct - percentage of sample to rserve for testing defaults to .20
                     return_one_df - return one big X, and y set for cross validation of entire sample
                     cols - columns to pull from sqldatabase
                     dependent_cutoffs - list of number of months to create buckets for dependent variable
                         default is [60] which will create two buckets (one <60 and one >= 60)

            returns: X_train, X_test, y_train, y_test, cols
                     X_train and X_test are pd.DataFrames, y_train and y_test are np.arrays
                     cols is a list of column names
                     if return_one_df, return one X, y
        """

        # pull specified fields from database using random rows.
        cols.append(dependent)
        df = super().load_data(source, cols, cond=self.where, sample_size=self.sample_size)
        df, dependent = self.clean_recode_data(df, dependent_cutoffs)

        # drop dependent colum from feature arrays
        y = df[dependent].values
        df = df.drop(dependent, 1)

        if return_one_df:
            return df, y

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_pct, random_state=0)

        return X_train, X_test, y_train, y_test, X_train.columns


    def test_models(self, 
                    source = 'breast', 
                    styles = [MultinomialNB, BernoulliNB, LinearRegression, KNeighborsRegressor, Lasso, Ridge], 
                    num_features = 3,
                    cols = ['YR_BRTH','AGE_DX','RACE','ORIGIN','LATERAL','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS'],
                    dependent_cutoffs=[60]):

        """ test_models(source = 'BREAST'):
            params:  source - table name in seer database, defaults to 'breast'
                     styles - list of classes to use to test the data i.e. [LinearRegression, LogisticRegression]
                              if styles is left empty, default routines in function will be used.
                              make sure to import modules containing the routines to test
                                i.e. from sklearn.linear_model import LinearRegression, LogisticRegression
                     num_features - number of features to test at one time, set to 99 to test all features in one run.
                     
            returns: n/a

            test various models against a combination of features, save scores to excel file named: source+'_seer_models.xlsx'
        """
        # name of excel file to dump results
        xls_name = source + '_seer_models.xlsx'
        # variable to predict
        dependent = 'SRV_TIME_MON'

        # get sets to train and test (80/20 split)
        X_train, X_test, y_train, y_test, cols = self.prepare_test_train_sets(source, dependent, test_pct = .20, cols = cols)
        col_cnt = len(cols)

        # make sure features to test is not greater than number of columns
        num_features = min(num_features, col_cnt)
        
        # formula for number of combinations: nCr = n! / r! (n - r)! 
        tot_cnt = (math.factorial(col_cnt) / (math.factorial(num_features) * math.factorial(col_cnt - num_features))) * len(styles)
        print("Processing: {0} tests.".format(int(tot_cnt)))

        res = []
        counter = 0
        for style in range(len(styles)):
            style_fnc = styles[style]
            model = style_fnc()
            print("Testing: {0}   ".format(style_fnc.__name__))
            for combo in itertools.combinations(cols, num_features):
                try: 
                    # train this model
                    X = X_train[list(combo)]
                    #X = preprocessing.scale(x)     # scale data if model requires it
                    model.fit(X, y_train)

                    # now test and score it
                    X = X_test[list(combo)]
                    #X = preprocessing.scale(x)     # scale data if model requires it

                    y_pred_test = model.predict(X)

                    y_pred_test = np.rint(y_pred_test)
                    y_pred_test = y_pred_test.astype(np.int)

                    f1 = f1_score(y_test, y_pred_test)
                    res.append([f1, style_fnc.__name__, [k for k in combo[:num_features]]])
                    counter += 1
                    if counter % 100 == 0:
                        print("Completed: {0}".format(counter, flush=True), end = '\r')
                except Exception as err:
                    counter += 1
                    if self.verbose:
                        print(err)
            del model

        # store trial results to excel
        res = sorted(res, reverse=True)
        res_df = pd.DataFrame(res)
        exc = pd.ExcelWriter(xls_name)
        res_df.to_excel(exc)
        exc.save()

        # cross validate and plot best model
        #TODO get parameters from res
        #self.cv_model(KNeighborsRegressor, 'KNeighborsRegressor', ['DATE_yr', 'ICDOTO9V', 'ICD_5DIG'])

        print("\nAll Completed: {0}  Results stored in: {1}".format(counter, xls_name))


    def clean_recode_data(self, df, dependent_cutoffs):
        """ clean_recode_data(df)
            params: df - dataframe of seer data to clean
            returns: cleaned dataframe, and name of new coded dependent variable

            Each cleaning step is on its own line so we can pick and choose what 
            steps we want after we decide on variables to study

            *** This is just a starting template.
            ***   I will finish when variables are determined.
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
            df = df[df.RADIATN < 7] 
        except Exception as err:
            pass

        try:
            # code as 1 or 2-more than one
            df.NUMPRIMS = df.NUMPRIMS.replace([x for x in range(2,37)], 2)
        except Exception as err:
            pass

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


        # creat new dependent column called SRV_BUCKET to hold the survival time value
        # based on the values sent into this function in the dependent_cutoffs list
        # first bucket is set to 0, next 1, etc...
        # Example dependent_cutoffs=[60,120,500]
        #   if survival is less than 60 SRV_BUCKET is set to 0
        #   if survival is >=60 and < 120 SRV_BUCKET is set to 1

        # create new column of all NaN
        df['SRV_BUCKET'] = np.NaN
        # fill buckets
        last_cut = 0       
        for x, cut in enumerate(dependent_cutoffs):
            df.loc[(df.SRV_TIME_MON >= last_cut) & (df.SRV_TIME_MON < cut), 'SRV_BUCKET'] = x
            last_cut = cut
        # assign all values larger than last cutoff to next bucket number       
        df['SRV_BUCKET'].fillna(len(dependent_cutoffs), inplace=True)

        df = df.drop('SRV_TIME_MON', 1)

        # categorical columns to one hot encode, check to make sure they are in df
        cat_cols_to_encode = list(set(['RACE', 'ORIGIN', 'SEX', 'TUMOR_2V', 'HISTREC']) & set(df.columns))
        df = self.one_hot_data(df, cat_cols_to_encode)

        df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        exc = pd.ExcelWriter('clean.xlsx')
        df.to_excel(exc)
        exc.save()

        return df, 'SRV_BUCKET'


    def one_hot_data(self, data, cols):
        """ Takes a dataframe and a list of columns that need to be encoded.
            Returns a new dataframe with the one hot encoded vectorized data
            
            See the following for explanation: 
                http://stackoverflow.com/questions/17469835/one-hot-encoding-for-machine-learning
            """
        # check to only encode columns that are in the data
        col_to_process = [c for c in cols if c in data]
        return pd.get_dummies(data, columns = col_to_process,  prefix = col_to_process)


    def cross_val_model(self, model, features, source='breast', sample_size=5000, num_folds = 5, dependent_cutoffs=[60]):
        """ cr_val_model(self, model, model_name, source)
            perform cross-validation on a specific model using specified sample size

            params: model - scikit-learn model function
                    features = list of features(fields) to use for model
                    source - table name in seer database, defaults to 'breast'
                    num_folds - number of folds for cross validation
                    dependent_cutoffs - list of number of months to create buckets for dependent variable
                        default is [60] which will create two buckets (one <60 and one >= 60)
        """
        
        mdl = model()
        model_name = model.__name__

        # variable to predict
        dependent = 'SRV_TIME_MON'

        # get all of the data, we will split to test/train using scikit's KFold routine
        X, y = self.prepare_test_train_sets(source, dependent, return_one_df=True, cols = features, dependent_cutoffs=dependent_cutoffs)
        X = np.array(X, dtype=np.float16)
        y = y.astype(np.int)

        kf = KFold(len(X), n_folds=num_folds, shuffle=True)
        # `means` will be a list of mean accuracies (one entry per fold)
        scores = {'precision':[], 'recall':[], 'f1':[]}
        for training, testing in kf:
            # Fit a model for this fold, then apply it to the
            Xtrn = X[training]
            #min_max_scaler = preprocessing.MinMaxScaler()
            #X_train_minmax = min_max_scaler.fit_transform(Xtrn)
            mdl.fit(Xtrn, y[training])

            Xtst = X[testing]
            #min_max_scaler = preprocessing.MinMaxScaler()
            #Xtst = min_max_scaler.fit_transform(Xtst)
            y_pred_test = mdl.predict(Xtst)
            y_pred_test = np.rint(y_pred_test)
            y_pred_test = y_pred_test.astype(np.int)

            # last batch is used for plotting
            y_test = y[testing]

            classificationReport = classification_report(y_test, y_pred_test)
            print("Report For: {0}".format(model_name))
            print(classificationReport)

            # Append scores for this run

            nn = precision_score(y_test, y_pred_test, average='micro')

            p,r,f,_ = precision_recall_fscore_support(y_test, y_pred_test)
            scores['precision'].append(p)
            scores['recall'].append(r)
            scores['f1'].append(f)

        # sort the y test data and keep the y_pred_test array in sync 
        # sort to make the graph more informative
        y_test, y_pred_test = zip(*sorted(zip(y_test, y_pred_test)))

        # plot last batch's results
        plt.plot([x for x in range(len(y_test))], y_pred_test, 'x', label="prediction")
        plt.plot([x for x in range(len(y_test))], y_test, 'o', label="data")
        plt.legend(loc='best')
        plt.title(model_name)
        # crop outliers so graph is more meaningful
        plt.ylim(0, 6)
        plt.show()

        #print(scores)


    def show_hist(self, df, cols):
        for col in cols:
            df.hist(col)
        plt.show()


if __name__ == '__main__':

    t0 = time.perf_counter()

    seer = ModelSeer(sample_size=1000, where="DATE_yr < 2008 AND O_DTH_CLASS = 0")
    
    ################ 

    # these three lines are used to display a histogram for the slected columns
    #_, df = seer.describe_data()
    #df = df[df.SRV_TIME_MON <= 360] 
    #seer.show_hist(df, ['SRV_TIME_MON'])

    ################ 

    # this line will run the selcted models
    #seer.test_models(styles = [RandomForestClassifier, KNeighborsRegressor, Lasso, Ridge], 
    #                 cols = ['YR_BRTH','AGE_DX','RACE','ORIGIN','LATERAL','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS'], num_features=3, dependent_cutoffs=[60, 120])

    ################ 

    # used to cross validate and plot a specific test and features.
    seer.cross_val_model(RandomForestClassifier, ['YR_BRTH','AGE_DX','RACE','ORIGIN','LATERAL','RADIATN','HISTREC','ERSTATUS','PRSTATUS','BEHANAL','HST_STGA','NUMPRIMS'], dependent_cutoffs=[60, 120])

    ################ 

    del seer
    print('\nModelSeer Module Elapsed Time: {0:.2f}'.format(time.perf_counter() - t0))