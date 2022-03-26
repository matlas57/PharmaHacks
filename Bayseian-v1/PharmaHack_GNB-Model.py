#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 11:41:33 2022

@author: williamzimmerman
"""

import pandas as pd
import sklearn as sk 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
'''
Global Variables for data use
'''


df = pd.read_csv("Gut_Microbiome_Data - challenge_1_gut_microbiome_data.csv")
size = df.shape

counts = [0,0,0,0]

'''
Use this function to enumerate the data for the disease#/healthy classification
Runs in O(n)
'''
def enumerateData():
    for i in range(0,size[0]):
        i
        val = df["disease"][i]
        val = val[-1]
        
        
        
        if(val == 'y'):
            val =0
        else:
            val =int(val)
        df["disease"][i]= val
        counts[val]+=1#to count instances of classes for Naive Bayes
    
   

def SplitandFit():
    y_train_n = []
    y_test_n = []
    X = df.columns[1:-2]
    Y = df[df.columns[-1]].copy()
    Y = Y.to_frame()
  
    X = df[X].copy()
   #X = X.to_frame()
   
    #print(X)
    #print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.99, random_state=0)
    y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()
    y_test = y_test.to_numpy()
    for i in y_train:
        y_train_n.append(i[0])
    
    y_train_n = np.asarray(y_train_n)
   
    print(y_train_n)

    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train_n)
    y_pred=y_pred.predict(X_test)
    
    
    for i in y_test:
        y_test_n.append(i[0])
    
    y_test_n = np.asarray(y_test_n)
   
    #print(y_pred)
    #print(y_test_n)
    print(accuracy_score(y_test_n, y_pred))
    

        
if __name__ == "__main__":
    enumerateData()
    SplitandFit()
    
    