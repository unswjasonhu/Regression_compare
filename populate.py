import pickle
import numpy as np
#from sklearn.preprocessing import MinMaxScaler

model_list = ['SVR','DTR','RFR','XGB','MLP','ABR','LR']

def load_model(modelname):
    """Load model using pickle"""
    fileObject = open(modelname,'rb')
    pipeline = pickle.load(fileObject)
    return pipeline

def main():
    """Populate estimates based on different regression models"""
    #Select different models from the list
    for model in model_list:
        clf = load_model(model)
        if model == 'MLP':
            #x = np.asarray([0.0/3,1.0/1,3.0/3,0,0,0.20000/0.6,0.20000/0.6,0.20000/0.6,0.30000/0.4])
            x = np.asarray([2.0/3,0.0/1,1.0/3,0,0,0.20000/0.6,0.10000/0.6,0.10000/0.6,0.20000/0.4])
            #x = np.asarray([0, 1, 3, 0, 0, 0.30000,0.30000,0.20000,0.40000])
            #x = np.asarray([2,0,1,0,0,0.30000,0.20000,0.10000,0.30000])
            x = x.reshape(1,9)
            y_val = np.zeros(shape=(100,100))
            for i in xrange(100):
                for j in xrange(100):
                    x[0][3] = (i-22)/(98.0-22)
                    x[0][4] = (j-18)/(98.0- 18)
                    y_val[i][j] = (clf.predict(x) *  5.74640) + 3.48652
            #Save estimates into a file
            np.savetxt('%s_populated_estimates.csv'% model, y_val,delimiter=',')
            print '%s estimates have been populated.' % model
        else:
            #Select a particular input
            #x = np.asarray([0, 1, 3, 0, 0, 0.30000,0.30000,0.20000,0.40000])
            x = np.asarray([2.0,0,1,0,0,0.30000,0.20000,0.10000,0.30000])
            x = x.reshape(1,9)
            #Create a numpy array to save estimates
            y_val = np.zeros(shape=(100,100))
            for i in xrange(100):
                for j in xrange(100):
                    x[0][3] = i
                    x[0][4] = j
                    y_val[i][j] = (clf.predict(x) *  5.74640) + 3.48652
            np.savetxt('%s_populated_estimates.csv'% model, y_val,delimiter=',')
            print '%s estimates have been populated.' % model


if __name__ == "__main__":
    print("Starting populate")
    main()
    print("Populating finished!")