#!/usr/bin/python

import pandas as pd
import numpy as np
import re
import pickle
import sys
import gzip

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

from theano import gof, config
from theano.gradient import grad, grad_not_implemented, DisconnectedType
import theano.tensor as T
import theano
from theano import shared

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

import xgboost as xgb

def loadData():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # try out with a subset
    #train = train.sample(n=40000)
    
    y = train.target
    train = train.drop('target',1)
    train = train.drop('ID',1)
    ids = test.ID.astype(str)
    test = test.drop('ID',1)
    
    # preprocess on some VARs
    train, test = preprocess(train, test)
    
    # scale it later

    # encode y
    le = LabelEncoder()
    y = le.fit_transform(y).astype(np.int32)

    pickle.dump((train, y, le), open('trainNoScale.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump((test, ids), open('testNoScale.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

def preprocess(train, test):
    print train.shape, test.shape
    # junk columns to delete
    # only one field
    count = train.apply(lambda x : len(x.unique()))
    count_1 = count.index[count == 1]
    train = train.drop(count_1, 1)
    test = test.drop(count_1, 1)
    print 'deleted: ', count_1
    #
    # too many nans
    fullOfNans = train.columns[np.sum(train.isnull()) > 145231*0.95]
    # make all the VAR list as pandas index type, easier for addition
    additional = pd.Index(['VAR_0227', 'VAR_0228'])
    train = train.drop(fullOfNans + additional, 1)
    test = test.drop(fullOfNans + additional, 1)
    print 'deleted: ', fullOfNans + additional
    #
    # duplicated
    count_2 = count.index[count == 2]
    train_count_2 = train[count_2]
    nan56 = train_count_2.columns[np.sum(train_count_2.isnull()) == 56]
    nan89 = train_count_2.columns[np.sum(train_count_2.isnull()) == 89]
    nan918 = train_count_2.columns[np.sum(train_count_2.isnull()) == 918]
    train = train.drop(nan56 + nan89 + nan918, 1)
    test = test.drop(nan56 + nan89 + nan918, 1)
    print 'deleted: ', nan56 + nan89 + nan918
    #
    # train set has three data types: int64, float64, O
    types = train.dtypes
    train_o = train[types.index[types == 'O']]
    test_o = test[types.index[types == 'O']]
    train_int = train[types.index[types == 'int64']]
    test_int = test[types.index[types == 'int64']]
    train_f = train[types.index[types == 'float64']]
    test_f = test[types.index[types == 'float64']]
    train = None
    test = None
    #
    # date featues
    isDate = train_o.apply(searchDates)    
    dates = train_o.columns[isDate]
    print 'parsing Date columns:'
    #
    trainDates = train_o[dates]
    testDates = test_o[dates]
    #
    train_o = train_o.drop(dates, 1)
    test_o = test_o.drop(dates, 1)
    #
    trainDates = parseDates(trainDates)
    testDates = parseDates(testDates)
    #
    print 'date features parsed.'
    #
    # other categorical features
    print 'parsing categorical columns'
    train_o = parseCat(train_o)
    test_o = parseCat(test_o)
    #
    # numerical features
    print 'parsing numerical columns'
    train_int = train_int.fillna(-99999999)  
    test_int = test_int.fillna(-99999999)  
    train_f = train_f.fillna(-99999999)  
    test_f = test_f.fillna(-99999999)
    #
    # combine
    train = pd.concat([trainDates, train_o, train_int, train_f],1)
    test = pd.concat([testDates, test_o, test_int, test_f],1)
    #
    return train, test

def searchDates(x):
    values = x.unique()
    trueVal = values[0]
    if pd.isnull(trueVal) and len(values) > 1:
        trueVal = values[1]
    return re.search(r'\d{2}[A-Z]{3}\d{2}', str(trueVal)) != None

def parseDates(df):    
    expanded = []
    for var in df.columns:   
        # fillna temperorily
        naPos = df[var].index[df[var].isnull()]
        df[var] = df[var].fillna('01Jan00:00:00:00')
        dtdf = df[var].map(lambda x : pd.datetime.strptime(x,"%d%b%y:%H:%M:%S"))
        year = dtdf.map(lambda x : x.year)
        month = dtdf.map(lambda x : x.month)
        day = dtdf.map(lambda x : x.day)
        hour = dtdf.map(lambda x : x.hour)
        expDate = pd.DataFrame([year,month,day,hour])
        expDate = expDate.T
        expDate.columns = [var+'_year',var+'_month',var+'_day',var+'_hour']
        #print expDate.shape
        # fillna with -1
        expDate.iloc[naPos] = expDate.iloc[naPos].applymap(lambda x : -1)
        expanded.append(expDate)
        print '\t%s parsed...' % var
    return pd.concat(expanded, 1)
     
def parseCat(df):
    print 'parsing categoricals...'
    df = df.fillna(-1)
    # you should use get_dummies ideally
    le = LabelEncoder()
    return df.apply(lambda x : le.fit_transform(x))
    
def ROC(pred, targets):
    print pred.ndim
    print targets.ndim
    roc = RocAucScoreOp()
    #print roc
    #exit()
    return roc(targets, pred)

class RocAucScoreOp(gof.Op):
    def __init__(self, name='roc_auc', use_c_code=theano.config.cxx):
        super(RocAucScoreOp, self).__init__(use_c_code)
        self.name = name
    def make_node(self, y_true, y_score):
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.scalar(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)
    def perform(self, node, inputs, out):
        y_true, y_score = inputs
        roc_auc = roc_auc_score(y_true, y_score)
        #y_out, = out
        #y = np.zeros_like(y_true[:, 0])
        #for i in xrange(len(y)):
        #    y[i] = T.as_tensor_variable(roc_auc)
        #y_out[0] = y
        out[0][0] = theano._asarray(roc_auc, dtype=config.floatX)
    def grad(self, inputs, grads):
        y_true, y_score = inputs
        g_y, = grads
        roc_grad = rocGrad()
        return [roc_grad(g_y, y_true, y_score),
                grad_not_implemented(self, 1, y_score)]
 
class rocGrad(gof.Op):
    __props__ = ()
    def make_node(self, g_y, coding_dist, true_one_of_n):
        return gof.Apply(self, [g_y, coding_dist, true_one_of_n],
                     [coding_dist.type()])
    def perform(self, node, inp, out):
        g_y, coding_dist, true_one_of_n = inp
        g_coding_strg, = out
        g_coding = np.zeros_like(coding_dist)
        for i in xrange(len(g_y)):
            g_coding[i, true_one_of_n[i]] = (-g_y[i] /
                                             coding_dist[i, true_one_of_n[i]])
        g_coding_strg[0] = g_coding
    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

class crossentropy(gof.Op):
    __props__ = ()
    def make_node(self, coding_dist, true_one_of_n):
        _coding_dist = T.as_tensor_variable(coding_dist)
        _true_one_of_n = T.as_tensor_variable(true_one_of_n)
        if _coding_dist.type.ndim != 2:
            raise TypeError('matrix required for argument: coding_dist')
        if _true_one_of_n.type not in (T.lvector, T.ivector):
            raise TypeError(
                'integer vector required for argument: true_one_of_n'
                '(got type: %s instead of: %s)' % (_true_one_of_n.type,
                                                   T.lvector))
        return gof.Apply(self, [_coding_dist, _true_one_of_n],
                     [T.Tensor(dtype=_coding_dist.dtype,
                      broadcastable=[False])()])
    def perform(self, node, inp, out):
        coding, one_of_n = inp
        y_out, = out
        y = np.zeros_like(coding[:, 0])
        for i in xrange(len(y)):
            y[i] = -np.log(coding[i, one_of_n[i]])
        y_out[0] = y
    def grad(self, inp, grads):
        coding, one_of_n = inp
        g_y, = grads
        crossentropy_categorical_1hot_grad = rocGrad()
        return [crossentropy_categorical_1hot_grad(g_y, coding, one_of_n),
                grad_not_implemented(self, 1, one_of_n)]    
 
def cat_crossentropy(pred, true):
    fun = crossentropy()
    return fun(pred, true)
   #print theano.T.nnet.categorical_crossentropy()
    #return theano.tensor.nnet.categorical_crossentropy(predictions, targets)
#--------------------------------------------------------------------------------------------------------
# Load data 
#loadData()
#exit()

# scale
#scaler = StandardScaler()
# default is np.float64, can't pickle
#train = scaler.fit_transform(train).astype(np.float32) 
#test = scaler.fit_transform(test).astype(np.float32)
#pickle.dump((train, y, le, scaler), open('trainNoScale.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
#pickle.dump((test, ids), open('testNoScale.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

#X, y, le, scaler = pickle.load(open('train.pkl'))
#X_test, ids = pickle.load(open('test.pkl'))

X, y, le = pickle.load(open('trainNoScale.pkl'))
X_test, ids = pickle.load(open('testNoScale.pkl'))

X = X.values
X_test = X_test.values
#print X.shape, len(y), len(le.classes_)
#print X_test.shape, len(ids)

if sys.argv[1] == 'nn':
    num_classes = len(le.classes_)
    num_features = X.shape[1]

    clf = NeuralNet(layers=[('input', InputLayer),
                         #('dense0', DenseLayer),
                         #('dropout0', DropoutLayer),
                         #('dense1', DenseLayer),
                         ('output', DenseLayer)],
                    input_shape=(None, num_features),
                    #dense0_num_units=100,
                    #dense0_nonlinearity=sigmoid,
                    #dropout0_p=0.1,
                    #dense1_num_units=100,

                    output_num_units=num_classes,
                    output_nonlinearity=softmax,

                    update=nesterov_momentum,
                    update_learning_rate=0.3,
                    update_momentum=0.8,
                     
                    #objective_loss_function = categorical_crossentropy,
                    objective_loss_function = ROC,  
                    #objective_loss_function = cat_crossentropy,
                     
                    train_split=TrainSplit(0.2),
                    verbose=1,
                    max_epochs=20)

    clf.fit(X, y)
    print roc_auc_score(y, clf.predict_proba(X)[:, 1])
    exit()

    preds = clf.predict_proba(X_test)[:, 1]
    with gzip.GzipFile('submit.csv.gz', mode='w') as gzfile:
        pd.DataFrame(preds, index=ids, columns=['target'])

if sys.argv[1] == 'xgb':
    param = { 
        'objective' : 'binary:logistic',        
        'max_depth' : 15, 
        'eta' : 0.1,
        'subsample' : 0.6,
        'colsample_bytree' : 0.6,
        #'alpha' : 0.0001,
        #'lambda' : 1,
        #'print.every.n' : 5, # no such feature for python
        'eval_metric' : 'auc'
        }
    num_round = 100

    kf = cross_validation.KFold(len(y), n_folds=4, random_state=33)
    for train_index, test_index in kf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)

        watchlist  = [(dtest,'eval'), (dtrain,'train')]
        clf = xgb.train(param, dtrain, num_round, watchlist)

        # check importance
        importance = clf.get_fscore()
        tuples = [(k, importance[k]) for k in importance]
        tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
        labels, values = zip(*tuples)
        print values
        for i in range(len(values)):
            if values[i] == 1:
                print labels[i], 
        print
            
        break
        #preds = clf.predict(dtest)









