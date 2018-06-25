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
from sklearn import preprocessing

def read_input():
    print("Loading all data...")
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/mo444_final_assignment/resources/recipe_boil_eff.csv')
    file_contents = pd.read_csv(file_path, dtype={'StyleID': np.int32}, delimiter = ';')
    
    #to_eliminate = [73, 164, 127, 16, 99, 110, 101, 154, 126, 104, 123, 46, 117, 96, 95, 79, 62, 3, 76, 133, 141, 125, 97, 166, 122, 64, 47, 130, 78, 48, 121, 161, 2, 74, 89, 139, 173, 172, 41, 140, 60, 55, 128, 158, 142, 17]
    restricted = [76, 142, 16, 46, 110, 127]
    
    indexes = list()
    
    print("Filtering data...")
    #Adjusting FG values
    for w in range(len(file_contents)):
        if file_contents.loc[w, 'FG'] > 1000:
            file_contents.loc[w, 'FG'] = file_contents.loc[w, 'FG']/1000
            
    #Eliminating classes with OG in Plato, examples with ABV = 0, BoilTime = 0 or 1 and Efficiency = 0 or 1
    for i in range (1, len(file_contents)):
        if (float(file_contents.loc[i][3]) == 0) | (float(file_contents.loc[i][5]) == 0) | (float(file_contents.loc[i][3]) == 1) | (float(file_contents.loc[i][6]) == 0) | (float(file_contents.loc[i][6]) == 1) | (float(file_contents.loc[i][0]) > 2):
            indexes.append(i)
    
    #Eliminate classes with only one occurrence
    for i in range (1, len(file_contents)):
        if (file_contents.loc[i, 'StyleID'] in restricted):
            if i not in indexes:
                indexes.append(i)
            
    #for i in range (1, len(file_contents)):
    #    if (file_contents.loc[i, 'StyleID'] in to_eliminate):
    #        if i not in indexes:
    #            indexes.append(i)
    
    for j in range(len(indexes)):
        file_contents = file_contents.drop(indexes[j])
    
    #Reset index count
    file_contents = file_contents.reset_index(drop=True)
    
    return file_contents

def read_input_wo():
    print("Loading all data...")
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/mo444_final_assignment/resources/recipe_boil_eff.csv')
    file_contents = pd.read_csv(file_path, dtype={'StyleID': np.int32}, delimiter = ';')
    
    #Classes with just one example
    restricted = [7, 10, 76, 142, 16, 46, 110, 127]
    
    indexes = list()
    
    print("Filtering data...")
    #Adjusting FG values
    for w in range(len(file_contents)):
        if file_contents.loc[w, 'FG'] > 1000:
            file_contents.loc[w, 'FG'] = file_contents.loc[w, 'FG']/1000
            
    #Eliminating classes with OG in Plato, examples with ABV = 0, BoilTime = 0 or 1 and Efficiency = 0 or 1
    for i in range (1, len(file_contents)):
        if (float(file_contents.loc[i][3]) == 0) | (float(file_contents.loc[i][5]) == 0) | (float(file_contents.loc[i][3]) == 1) | (float(file_contents.loc[i][6]) == 0) | (float(file_contents.loc[i][6]) == 1) | (float(file_contents.loc[i][0]) > 2):
            indexes.append(i)
    
    #Eliminate classes with only one occurrence
    for i in range (1, len(file_contents)):
        if (file_contents.loc[i, 'StyleID'] in restricted):
            if i not in indexes:
                indexes.append(i)
            
    #for i in range (1, len(file_contents)):
    #    if (file_contents.loc[i, 'StyleID'] in to_eliminate):
    #        if i not in indexes:
    #            indexes.append(i)
    
    for j in range(len(indexes)):
        file_contents = file_contents.drop(indexes[j])
    
    #Reset index count
    file_contents = file_contents.reset_index(drop=True)
    
    return file_contents

#Separates classes into two groups for better classification
def separate_groups(file_contents, threshold):
    
    #Threshold number of occurrences
    separation_threshold = threshold
    
    group_1_classes = list()
    group_2_classes = list()
    
    #Sort classes by frequency and put it in hash
    sorted_classes_dict = file_contents['StyleID'].value_counts().to_dict()
    
    #Iterate through hash separating glasses in 2 groups
    for key, value in sorted_classes_dict.items():
        if value >= separation_threshold:
            group_1_classes.append(key)
        else:
            if value > 1: 
                group_2_classes.append(key)

    #Make 2 copies of original dataset
    dataset_1 = file_contents.copy()
    dataset_2 = file_contents.copy()
    
    #Create empty lists that will store indexes in each dataset to be eliminated
    purge_index_1 = list()
    purge_index_2 = list()
    
    #Get indexes that doesn't belong to dataset_1
    for i in range (0, len(dataset_1)):
        if (file_contents.loc[i, 'StyleID'] not in group_1_classes):
            if i not in purge_index_1:
                purge_index_1.append(i)

    #Get indexes that doesn't belong to dataset_2
    for i in range (0, len(dataset_2)):
        if (file_contents.loc[i, 'StyleID'] not in group_2_classes):
            if i not in purge_index_2:
                purge_index_2.append(i)
            
    #Strip unwanted indexes from dataset_1
    for j in range(len(purge_index_1)):
        dataset_1 = dataset_1.drop(purge_index_1[j])
        
    #Strip unwanted indexes from dataset_2
    for j in range(len(purge_index_2)):
        dataset_2 = dataset_2.drop(purge_index_2[j])
            
    #dataset_2 = dataset_2.drop(0)        
        
    #For each dataset, reset index count
    dataset_1 = dataset_1.reset_index(drop=True)
    dataset_2 = dataset_2.reset_index(drop=True)

    return dataset_1, dataset_2

def count_elements(file_contents):
    elements = {}
    for elem in file_contents:
        if elem in elements.keys():
            elements[elem] += 1
        else:
            elements[elem] = 1
    return elements

#Split dataset into balanced train and test sets
def test_train_split(file_contents):
    y = file_contents['StyleID']
    X = file_contents[file_contents.columns[0:5]]
    
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
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
def knn(X_train, y_train, X_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    pred2 = knn.predict(X_train)
    
    #Validation
    print (accuracy_score(y_test, pred))
    #Train
    print (accuracy_score(y_train, pred2))

#Multiclass SVM (OVO)
def multiclass_svm_ovo(X_train, y_train, X_test, y_test):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    #clf.decision_function(X_train)
    
    pred = clf.predict(X_test)
    pred2 = clf.predict(X_train)

    #Validation
    print (accuracy_score(y_test, pred))
    #Train
    print (accuracy_score(y_train, pred2))

#Multiclass SVM (OVA)
def multiclass_svm_ova(X_train, y_train, X_test, y_test):
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train) 
    pred = lin_clf.predict(X_test)
    
    print(accuracy_score(y_test, pred))

    #clf.decision_function(X_train)

#Random Forests
def random_forest(X_train, y_train, X_test, y_test, n_estimators):
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, random_state=123456, min_samples_leaf = 4, max_depth = 15)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    pred2 = rf.predict(X_train)
    
    #Validation
    print(accuracy_score(y_test, pred))
    #Train
    print(accuracy_score(y_train, pred2))


#Confusion matrix
def plot_confusion_matrix(y, y_test, pred):
    cm = pd.DataFrame(confusion_matrix(y_test, pred), columns=set(y_test), index=set(y_test))
    plt.show(sns.heatmap(cm, annot=True))
    

def main():
    
    file_contents = read_input()
    
    #First approach
    X_train, y_train, X_test, y_test = test_train_split(file_contents)
    knn(X_train, y_train, X_test, y_test, 50)
    multiclass_svm_ovo(X_train, y_train, X_test, y_test)
    #Grid search for random forests
    random_forest(X_train, y_train, X_test, y_test, 50)
    random_forest(X_train, y_train, X_test, y_test, 100)
    random_forest(X_train, y_train, X_test, y_test, 200)
    random_forest(X_train, y_train, X_test, y_test, 300)
    random_forest(X_train, y_train, X_test, y_test, 400)
    random_forest(X_train, y_train, X_test, y_test, 500)
    
    #Second approach
    split = 200
    if split <= 900:
        dataset_1, dataset_2 = separate_groups(file_contents, split)
        X_train, y_train, X_test, y_test = test_train_split(dataset_1)
        random_forest(X_train, y_train, X_test, y_test, 200)
        
        X_train, y_train, X_test, y_test = test_train_split(dataset_2)
        multiclass_svm_ovo(X_train, y_train, X_test, y_test)
        random_forest(X_train, y_train, X_test, y_test, 200)
        split = split + 50
    
