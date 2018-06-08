#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:28:51 2018

@author: Pedro
"""
import os
import csv
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def read_input():
    print("Loading all data...")
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/mo444_final_assignment/resources/recipe_mod2.csv')
    file_contents = pd.read_csv(file_path, delimiter = ';')
    
    indexes = list()
    
    print("Filtering data...")
    
    #Adjusting FG values
    for w in range(len(file_contents)):
        if file_contents.loc[w, 'FG'] > 1000:
            file_contents.loc[w, 'FG'] = file_contents.loc[w, 'FG']/1000
            
    #Eliminating classes with only one example, examples with OG in Plato and examples with ABV = 0
    for i in range (1, len(file_contents)):
        if (float(file_contents.loc[i][3]) == 0) | (float(file_contents.loc[i][0]) > 2) | file_contents.loc[i, 'StyleID'] == 142 | file_contents.loc[i, 'StyleID'] == 16 | file_contents.loc[i, 'StyleID'] == 46 | file_contents.loc[i, 'StyleID'] == 110 | file_contents.loc[i, 'StyleID'] == 127 | file_contents.loc[i, 'StyleID'] == 177:
            indexes.append(i)
    
    for j in range(len(indexes)):
        file_contents = file_contents.drop(indexes[j])
    
    
    return file_contents

#Split dataset into balanced train and test sets
def test_train_split(file_contents):
    y = file_contents['StyleID']
    X = file_contents[file_contents.columns[0:5]]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    generator = sss.split(X, y)
    gen_list = list(generator)
    
    train_index = gen_list[0][0]
    test_index = gen_list[0][1]
    
    X_train = list()
    y_train = list()
    X_test = list()
    y_test = list()
    
    for index in train_index:
        X_train.append(X.loc[index])
        y_train.append(y.loc[index])
        
    for index in test_index:
        X_test.append(X.loc[index])
        y_test.append(y.loc[index])

    return X_train, y_train, X_test, y_test
    
#K-Nearest-Neighbors    
def knn(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=176)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    
    #Yields an accuracy of 31%
    print (accuracy_score(y_test, pred))

#Multiclass SVM (OVO)
def multiclass_svm_ovo(X_train, y_train, X_test, y_test):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    #clf.decision_function(X_train)
    
    pred = clf.predict(X_test)
    
    #Yields an accuracy of 33%
    print (accuracy_score(y_test, pred))

#Multiclass SVM (OVA)
def multiclass_svm_ova(X_train, y_train, X_test, y_test):
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train) 
    pred = lin_clf.predict(X_test)
    
    #Yields an accuracy of 3.1%
    print(accuracy_score(y_test, pred))

    #clf.decision_function(X_train)

#Random Forests
def random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=123456)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    
    #Yields an accuracy of 32.5% with n_estimators = 100 and 33% with n_estimators = 300
    print(accuracy_score(y_test, pred))


def plot_confusion_matrix(y, y_test, pred):
    cm = pd.DataFrame(confusion_matrix(y_test, pred), columns=set(y_test), index=set(y_test))
    plt.show(sns.heatmap(cm, annot=True))
    
    
def main():
    
    file_contents = read_input()
    
    return file_contents
