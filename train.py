import numpy as np
import pandas as pd
import pickle
import os
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import *
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor

path = os.getcwd()
df = pd.read_csv(path + '/data.csv', sep = ',')
#Select model to use below
use_SVR = 1
use_LR = 1
use_MLP = 1
use_XGB = 1
use_RFR = 1
use_PLSR = 1
use_DTR = 1

def classify_hour(hour):
    """ Classify the hour based on some aggregate ranges"""
    #Aggregate hours into 4 classes, 8-11/12-15/16-19/20-23
    if hour >=8 and hour  <= 11:
        return 1
    elif hour >= 12 and hour <= 15:
        return 2
    elif hour >= 16 and hour <= 19:
        return 3
    elif hour >= 20 and hour <= 23:
        return 4
    raise Exception("Hour out of bounds")

def predict(model,input,output,kf,dataframe,modelname):
    """Train the model and save estimates"""
    #Get mean absolute error and mean squared error, and save them into a file
    MAE = cross_val_score(model, input, output, cv=kf, n_jobs = 1,scoring= 'mean_absolute_error')
    MSE = cross_val_score(model, input, output, cv=kf, n_jobs = 1,scoring= 'mean_squared_error')
    parameter = open(path + '/parameters.txt','a')
    parameter.write('%s MAE and MSE is %f and %f respectively\n' % \
                    (modelname, -sum(MAE) / float(len(MAE)),-sum(MSE) / float(len(MSE))))
    parameter.close()
    print "Saved the %s model parameters." % modelname
    #Get estimates and save them into a csv file
    predicted = cross_val_predict(model, input, output, cv=kf, n_jobs = 1)
    dataframe[modelname] = pd.DataFrame(np.asarray(predicted))
    dataframe.to_csv(path+ '/estimate.csv',sep=',')
    #Dump the model into a file use pickle
    pipeline_set = model.fit(input,output)
    file_name = modelname
    fileObject = open(path + '/' + file_name,'wb')
    pickle.dump(pipeline_set, fileObject)
    fileObject.close()
    print "Pickled the %s model." % modelname

def main():
    """Train different models"""
    X = []
    y = []
    #Prepare training data
    for i in range(df.shape[0]):
        df.loc[i,'timerange'] = classify_hour(df.loc[i,'time'])
        X.append([df.loc[i,'timerange'], df.loc[i,'weekdays'], df.loc[i,'season'], df.loc[i,'grid_location_row'], \
                  df.loc[i,'grid_location_col'], df.loc[i,'co_chullora'], df.loc[i,'co_liverpool'], df.loc[i,'co_prospect'],\
                  df.loc[i,'co_rozelle']])
        y.append(df.loc[i,'co'])        
    x = np.float64(X)
    y = np.float64(y)
    #Split the data into 10 folds
    kf_total = KFold(df.shape[0] ,n_folds=10, shuffle=True, random_state=0)
    #Train different models
    if use_SVR:
        clf = SVR(C=1.0,epsilon=0.2,gamma='auto',kernel='rbf',verbose=True,cache_size=3000)
        predict(clf, x, y, kf_total, df, 'SVR')
    if use_LR:
        clf = LinearRegression(fit_intercept=True, normalize=False)
        predict(clf, x, y, kf_total, df, 'LR')
    if use_MLP:
        clf = MLPRegressor(hidden_layer_sizes = 150, learning_rate_init=0.001, max_iter=500)
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        predict(clf, x, y, kf_total, df, 'MLP')
    if use_XGB:
        clf = xgb.XGBRegressor()
        predict(clf, x, y, kf_total, df, 'XGB')
    if use_RFR:
        clf = RandomForestRegressor(min_samples_leaf = 20)
        predict(clf, x, y, kf_total, df, 'RFR')
    if use_PLSR:
        clf = PLSRegression()
        predict(clf, x, y, kf_total, df, 'PLSR')
    if use_DTR:
        clf = DecisionTreeRegressor(max_depth=500,min_samples_leaf =20)
        predict(clf, x, y, kf_total, df, 'DTR')


if __name__ == "__main__":
    """Start the scripts."""
    print 'Training start.'
    main()
    print 'Training end.'
