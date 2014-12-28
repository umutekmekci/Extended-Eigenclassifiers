# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:52:49 2014

@author: daredavil
"""
import h5py
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import cross_val_score

def read_text(file_name):
    X_train = []
    with open(file_name,'r') as f_:
        for line in f_:
            X_train.append(map(float,line.strip().split(',')))
    return np.array(X_train)
        
def shape_data(X_train):
    return np.hstack((X_train[:,:2],X_train[:,-1][:,np.newaxis],X_train[:,2:-1]))
        
def compress(row):
    return (row - row.min())/(row.max()-row.min())
    
def read_lambdas(file_name):
    with h5py.File(file_name,'r') as f:
        X1 = np.array(f[u'X1'])
        X2 = np.array(f[u'X2'])
    return X1,X2
    
    
if __name__ == '__main__':
    file_name = 'precsMat2.txt'
    y = read_text(file_name)
    y = shape_data(y)
    y = np.array([compress(row) for row in y])
    y_ind = (y >= 0.8)
    y = np.zeros_like(y)
    y[y_ind] = 1
    #print X_train[0]
    #print
    #print X_train[1]
    X1,X2 = read_lambdas(r'C:\Users\daredavil\Documents\MATLAB\Fusion\matfiles\FUSSELECTPYTHON.mat')
    loo=LeaveOneOut(y.shape[0])
    clf = DecisionTreeClassifier(criterion='entropy',random_state = 500)
    scores = cross_val_score(clf,X1,y,cv = loo,scoring = lambda est,X,y: np.sum(np.maximum(0,y-est.predict(X))))
    print scores.mean()    
    #clf = RandomForestClassifier(criterion='entropy',random_state = 500,n_estimators=5)
    #scores = cross_val_score(clf,X1,y,cv = loo,scoring = lambda est,X,y: np.sum(np.maximum(0,y-est.predict(X))))
    #print scores.mean()
    clf = DecisionTreeClassifier(criterion='entropy',random_state = 500, max_depth = 15)
    clf.fit(X1,y)
    print clf.feature_importances_
    #raise
    #y_pred = clf.predict(X1)
    #print y_pred
    #print y
    #print np.sum(np.abs(y-y_pred))
    #from sklearn.externals.six import StringIO 
    #import pydot
    #dot_data=StringIO()
	
    with open("tree.dot", 'w') as f:
		tree.export_graphviz(clf, out_file=f) 
    
	#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    #graph.write_pdf("tree_vis.pdf") 