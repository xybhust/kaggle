# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:02:42 2016

@author: Yibing
"""
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer

import mlp
import logistic_sgd as ls
import cnn



if __name__ == "__main__":

     
    # ================== SVM ===============================
#    classif = OneVsRestClassifier(SVC(C=1., kernel='linear'))
#    classif.fit(X_train, y_train)
#    print classif.score(X_train, y_train)
#    print classif.score(X_test, y_test)
    
    # =================== Logistic Regression ================  
#    classif = OneVsRestClassifier(LogisticRegression())
#    classif.fit(X_train, y_train)
#    print classif.score(X_train, y_train)
#    print classif.score(X_test, y_test)

    # =================== MLP ==================================
    #mlp.test_mlp('data/X.csv', 'data/y.csv')
    #mlp.predict('best_mlp.pkl', 'data/X_pred.csv')
   # cnn.evaluate_lenet5(file_x='data/X.csv', file_y='data/y.csv')
    cnn.predict('best_cnn.pkl', 'data/X_pred.csv')
