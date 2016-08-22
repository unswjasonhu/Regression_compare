import pandas as pd
pd.options.display.max_columns = 100
from sklearn.cross_validation import train_test_split
pd.options.display.max_rows = 100
from copy import deepcopy
from math import sqrt
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cross_validation import cross_val_score, KFold, cross_val_predict

def classify_hour(hour):
    """ Classify the hour based on some aggregate ranges"""
    #Aggregate hours into 4 classes, 8-11/12-15/16-19/20-23
    if hour >=8 and hour  <= 11:
        return 0
    elif hour >= 12 and hour <= 15:
        return 1
    elif hour >= 16 and hour <= 19:
        return 2
    elif hour >= 20 and hour <= 23:
        return 3
    raise Exception("Hour out of bounds")

def split_data(file):
    df = pd.read_csv(file,sep = ',')
    for i in range(df.shape[0]):
        df.loc[i,'time_range'] = classify_hour(df.loc[i,'time'])
    train, test = train_test_split(df, test_size= 0.1)
    train.to_csv('train_'+ file, sep = ',',index= False)
    test.to_csv('test_'+ file, sep = ',', index= False)
    return df, pd.read_csv('train_'+ file, sep = ','), pd.read_csv('test_'+ file, sep = ',')

def remove_feature(dataset):
    dataset.drop(['datetime','date','time','dayoftheweek','co_original','co_mean','co_stddev'], axis=1,inplace = True)
    return dataset.reindex_axis(['time_range'] + list(dataset.columns[:-1]),axis = 1)

def scale_all_features(dataset):
    features = list(dataset.columns)
    features.remove('co')
    #print dataset[features].max()
    #print dataset[features].min()
    dataset[features] = dataset[features].apply(lambda x: (x.astype('float')- x.min())/(x.max() - x.min()), axis=0)
    return dataset


def feature_engineering(filename):
    total_set, train_set, test_set = split_data(filename)
    total_set1 = remove_feature(deepcopy(total_set))
    total_set2 = remove_feature(deepcopy(total_set))
    train_set1 = remove_feature(deepcopy(train_set))
    train_set2 = remove_feature(deepcopy(train_set))
    test_set1  = remove_feature(deepcopy(test_set))
    test_set2 = remove_feature(deepcopy(test_set))
    #data['time_range'].fillna(data['time_range'].median(), inplace=True)
    #season_co= train_set.groupby('season')['co'].mean()
    #hour_co = train_set.groupby('time_range')['co'].mean()
    #train_set[train_set['season'] == 2]['season'].value_counts()
    #season_dummies = pd.get_dummies(total_set['season'],prefix='Season')
    #total_set= pd.concat([total_set,season_dummies],axis = 1)
    #combined['time'] = combined['season'].map({'0':'Summer','1':'Autumn'})
    return scale_all_features(total_set1), scale_all_features(train_set1),scale_all_features(test_set1), total_set2, \
           train_set2, test_set2

def cross_validation(model,input,output,kf,modelname):
    """Train the model and save estimates"""
    #Get mean absolute error and root mean squared error, and save them into a file
    MAE = cross_val_score(model, input, output, cv=kf,scoring= 'mean_absolute_error')
    RMSE = (-cross_val_score(model, input, output, cv=kf,scoring= 'mean_squared_error')) ** 0.5
    parameter = open('parameters.txt','a')
    parameter.write('%s MAE and RMSE for whole dataset is %f and %f respectively\n' % \
                    (modelname, -sum(MAE) / float(len(MAE)),sum(RMSE) / float(len(RMSE))))
    parameter.close()
    print "Saved the %s model parameters." % modelname
    #Get estimates and save them into a csv file
    #predicted = cross_val_predict(model, input, output, cv=kf)
    #dataframe[modelname] = pd.DataFrame(np.asarray(predicted))
    #dataframe.to_csv(path+ '/estimate.csv',sep=',')

def save_model(model,input,output,modelname):
    #Dump the model into a file use pickle
    pipeline_set = model.fit(input.as_matrix(),output.tolist())
    file_name = modelname
    fileObject = open(file_name,'wb')
    pickle.dump(pipeline_set, fileObject)
    fileObject.close()
    print "Pickled the %s model." % modelname

def test(modelname, test_input, test_output,df):
    fileObject = open(modelname,'rb')
    pipeline = pickle.load(fileObject)
    predict = pipeline.predict(test_input.as_matrix())
    MAE = mean_absolute_error(test_output, predict)
    RMSE = sqrt(mean_squared_error(test_output, predict))
    parameter = open('parameters.txt','a')
    parameter.write('%s MAE and RMSE for testing set is %f and %f respectively\n' % \
                    (modelname,MAE, RMSE))
    parameter.close()
    df[modelname] = (pd.DataFrame(predict) *  5.74640) + 3.48652
    df.to_csv('estimate.csv',sep=',')


