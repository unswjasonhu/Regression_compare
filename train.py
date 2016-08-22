from Feature_engineering import *
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score, KFold, cross_val_predict
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostRegressor


#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.feature_selection import SelectFromModel

use_model = ['SVR', 'DTR', 'RFR', 'XGB', 'MLP', 'ABR', 'LR']


def main():
    total_set, train_set, test_set, total_set_nonscale, train_set_nonscale, test_set_nonscale = feature_engineering('data.csv')
    total_set_input = total_set.drop('co',axis = 1)
    total_set_output= total_set.co
    train_set_input = train_set.drop('co',axis = 1)
    train_set_output = train_set.co
    test_set_input = test_set.drop('co',axis = 1)
    test_set_output = test_set.co
    total_set_nonscale_input = total_set_nonscale.drop('co',axis = 1)
    total_set_nonscale_output = total_set_nonscale.co
    train_set_nonscale_input = train_set_nonscale.drop('co',axis = 1)
    train_set_nonscale_output = train_set_nonscale.co
    test_set_nonscale_input = test_set_nonscale.drop('co',axis = 1)
    test_set_nonscale_output = test_set_nonscale.co
    kf_total = KFold(total_set_input.shape[0],n_folds=10, shuffle=True, random_state=0)
    kf_train = KFold(train_set_input.shape[0],n_folds=10, shuffle=True, random_state=0)
    #Train different models
    df = pd.DataFrame(test_set_output,columns=['co'])
    df['co_original'] = (df['co'] * 5.74640) + 3.48652
    for model in use_model:
        if model == 'MLP':
            clf = MLPRegressor(hidden_layer_sizes = 150, learning_rate_init=0.001, max_iter=500)
            cross_validation(clf, total_set_input, total_set_output,  kf_total, model)
            save_model(clf,train_set_input, train_set_output,model)
            test(model, test_set_input,test_set_output,df)
        else:
            if model == 'SVR':
                clf = SVR(C=1.0,epsilon=0.2,gamma='auto',kernel='rbf',verbose=True,cache_size=3000)
            elif model == 'LR':
                clf = LinearRegression(fit_intercept=True, normalize=False)
            elif model == 'XGB':
                clf = xgb.XGBRegressor()
            elif model == 'RFR':
                clf = RandomForestRegressor(min_samples_leaf = 10)
            elif model == 'DTR':
                clf = DecisionTreeRegressor(max_depth=500,min_samples_leaf = 10)
            elif model == 'ABR':
                clf = AdaBoostRegressor()
            else:
                print 'Wrong model!'
            cross_validation(clf, total_set_nonscale_input, total_set_nonscale_output, kf_total, model)
            save_model(clf, train_set_nonscale_input, train_set_nonscale_output, model)
            test(model,test_set_nonscale_input, test_set_nonscale_output, df)

if __name__ == "__main__":
    print 'Training Start!'
    main()
    print 'Done!'


