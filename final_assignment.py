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
    file_path = Path('/Users/Pedro/Library/Mobile Documents/com~apple~CloudDocs/MO444/Final Assignment/beer-recipes/recipeData.csv')
    file_contents = pd.read_csv(file_path, header=None)
    
    return file_contents     


def main():
    
    file_contents = read_input()
    
    return file_contents
