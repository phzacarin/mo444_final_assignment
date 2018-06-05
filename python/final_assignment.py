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
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize

def read_input():
    print("Loading all data...")
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/mo444_final_assignment/resources/filtered_data.csv')
    #file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/mo444_final_assignment/resources/recipe_mod2.csv')
    file_contents = pd.read_csv(file_path, header=None, delimiter = ';')
    
    #contents_filtered = file_contents.copy()
    indexes = list()
    
    print("Filtering data...")
    for i in range (1, len(file_contents)):
        if (float(file_contents.loc[i,3]) == 0) | (float(file_contents.loc[i,0]) > 2):
            indexes.append(i)
    
    for j in range(len(indexes)):
        file_contents = file_contents.drop(indexes[j])
        
        
    return file_contents


def main():
    
    file_contents = read_input()
    
    return file_contents
