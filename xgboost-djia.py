import pandas as pd
import numpy as np
import warnings
import tkinter
from matplotlib import pyplot
#from pandas import read_csv, set_option
from pandas import Series, datetime
from pandas.tools.plotting import scatter_matrix, autocorrelation_plot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBClassifier
import seaborn as sns

sns.set()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    sentence_file = "data/combined_stock_data.csv"
    sentence_df = pd.read_csv(sentence_file, parse_dates=[1])
    #print(sentence_df.head())
    #print(sentence_df.shape)
    #print(sentence_df.dtypes)

    # Load the stock prices dataset into a dataframe and check the top 5 rows
    stock_prices = "stocknews/DJIA_table.csv"
    stock_data = pd.read_csv(stock_prices, parse_dates=[0])
    stock_data = stock_data.reindex(index=stock_data.index[::-1])


    # Create a dataframe by merging the headlines and the stock prices dataframe
    merged_dataframe = sentence_df[['Date', 'Label', 'Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']].merge(stock_data, how='inner', on='Date', left_index=True)
    # Check the shape and top 5 rows of the merged dataframe

    # Push the Label column to the end of the dataframe
    cols = list(merged_dataframe)
    print(cols)
    cols.append(cols.pop(cols.index('Label')))
    merged_dataframe = merged_dataframe.ix[:, cols]

    # Change the datatype of the volume column to float
    #merged_dataframe['Date'] = pd.to_datetime(merged_dataframe['Date'])
    merged_dataframe['Volume'] = merged_dataframe['Volume'].astype(float)
    #merged_dataframe = merged_dataframe.set_index(['Date'])
    merged_dataframe.index = merged_dataframe.index.sort_values()

    #print(merged_dataframe.describe())

    '''merged_dataframe.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(10, 10))
    pyplot.show()
    pyplot.scatter(merged_dataframe['Subjectivity'], merged_dataframe['Label'])
    pyplot.xlabel('Subjectivity')
    pyplot.ylabel('Stock Price Up or Down 0: Down, 1: Up')
    pyplot.show()
    pyplot.scatter(merged_dataframe['Objectivity'], merged_dataframe['Label'])
    pyplot.xlabel('Objectivity')
    pyplot.ylabel('Stock Price Up or Down 0: Down, 1: Up')
    pyplot.show()
    merged_dataframe['Subjectivity'].plot('hist')
    pyplot.xlabel('Subjectivity')
    pyplot.ylabel('Frequency')
    pyplot.show()
    merged_dataframe['Objectivity'].plot('hist')
    pyplot.xlabel('Subjectivity')
    pyplot.ylabel('Frequency')
    pyplot.show()
    print("Size of the Labels column")
    print(merged_dataframe.groupby('Label').size())
    '''
    '''
    md_copy = merged_dataframe
    md_copy = md_copy.replace(-1, np.NaN)
    import missingno as msno
    # Nullity or missing values by columns
    msno.matrix(df=md_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))

    colormap = pyplot.cm.afmhot
    pyplot.figure(figsize=(16,12))
    pyplot.title('Pearson correlation of continuous features', y=1.05, size=15)
    sns.heatmap(merged_dataframe.corr(),linewidths=0.1,vmax=1.0, square=True,
                cmap=colormap, linecolor='white', annot=True)
    pyplot.show()'''


    # Print the datatypes and count of the dataframe
    print(merged_dataframe.dtypes)
    print(merged_dataframe.count())
    # Change the NaN values to the mean value of that column
    nan_list = ['Subjectivity', 'Objectivity', 'Positive', 'Negative', 'Neutral']
    for col in nan_list:
        merged_dataframe[col] = merged_dataframe[col].fillna(merged_dataframe[col].mean())

    # Recheck the count
    print(merged_dataframe.count())
    # Separate the dataframe for input(X) and output variables(y)
    X = merged_dataframe.loc[:,'Subjectivity':'Adj Close']
    y = merged_dataframe.loc[:,'Label']
    # Set the validation size, i.e the test set to 20%
    validation_size = 0.20
    # Split the dataset to test and train sets
    # Split the initial 70% of the data as training set and the remaining 30% data as the testing set
    train_size = int(len(X.index) * 0.7)
    val_size = int(train_size*1)

    X_train, X_test = X.loc[0:train_size-1, :], X.loc[train_size-1: len(X.index)-2, :]
    y_train, y_test = y.loc[1:train_size], y.loc[train_size: len(y.index)]
    print(len(X), len(y))
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    print('Observations: %d' % (len(X.index)))
    print('X Training Observations: %d' % (len(X_train.index)))
    print('X Testing Observations: %d' % (len(X_test.index)))
    print('y Training Observations: %d' % (len(y_train)))
    print('y Testing Observations: %d' % (len(y_test)))
    #pyplot.plot(X_train['Objectivity'])
    #pyplot.plot([None for i in X_train['Objectivity']] + [x for x in X_test['Objectivity']])
    #pyplot.show()
    num_folds = 10
    scoring = 'accuracy'
    # Append the models to the models list
    models = []
    models.append(('LR' , LogisticRegression()))
    models.append(('LDA' , LinearDiscriminantAnalysis()))
    models.append(('KNN' , KNeighborsClassifier()))
    models.append(('CART' , DecisionTreeClassifier()))
    models.append(('NB' , GaussianNB()))
    models.append(('SVM' , SVC()))
    models.append(('RF' , RandomForestClassifier(n_estimators=50)))
    models.append(('XGBoost', XGBClassifier()))

    # Evaluate each algorithm for accuracy
    results = []
    names = []

    for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accu_score = accuracy_score(y_test, y_pred)
        print(name + ": " + str(accu_score))


    # prepare the model LDA
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(rescaledX, y_train)
    # estimate accuracy on validation dataset
    rescaledValidationX = scaler.transform(X_test)
    predictions = model_lda.predict(rescaledValidationX)
    print("accuracy score:")
    print(accuracy_score(y_test, predictions))
    print("confusion matrix: ")
    print(confusion_matrix(y_test, predictions))
    print("classification report: ")
    print(classification_report(y_test, predictions))

    '''
    model_xgb = XGBClassifier()
    model_xgb.fit(rescaledX, y_train)
    # estimate accuracy on validation dataset
    rescaledValidationX = scaler.transform(X_test)
    predictions = model_xgb.predict(rescaledValidationX)
    print("accuracy score:")
    print(accuracy_score(y_test, predictions))
    print("confusion matrix: ")
    print(confusion_matrix(y_test, predictions))
    print("classification report: ")
    print(classification_report(y_test, predictions))

    # Scaling Random Forests

    model_rf = RandomForestClassifier(n_estimators=1000)
    model_rf.fit(rescaledX, y_train)
    # estimate accuracy on validation dataset
    rescaledValidationX = scaler.transform(X_test)
    predictions = model_rf.predict(rescaledValidationX)
    print("accuracy score:")
    print(accuracy_score(y_test, predictions))
    print("confusion matrix: ")
    print(confusion_matrix(y_test, predictions))
    print("classification report: ")
    print(classification_report(y_test, predictions))

    # XGBoost on Stock Price dataset, Tune n_estimators and max_depth
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    import matplotlib

    matplotlib.use('Agg')
    model = XGBClassifier()
    n_estimators = [150, 200, 250, 450, 500, 550, 1000]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print(max_depth)
    best_depth = 0
    best_estimator = 0
    max_score = 0
    for n in n_estimators:
        for md in max_depth:
            model = XGBClassifier(n_estimators=n, max_depth=md)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            if score > max_score:
                max_score = score
                best_depth = md
                best_estimator = n
            print("Score is " + str(score) + " at depth of " + str(md) + " and estimator " + str(n))
    print("Best score is " + str(max_score) + " at depth of " + str(best_depth) + " and estimator of " + str(best_estimator))
    '''
