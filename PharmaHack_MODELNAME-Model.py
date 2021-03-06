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
'''
Global Variables for data use
'''
df = pd.read_csv("/Users/williamzimmerman/Desktop/PharmaHacks/Gut_Microbiome_Data - challenge_1_gut_microbiome_data.csv")
size = df.shape
print(df.head)


'''
Use this function to enumerate the data for the disease#/healthy classification
Runs in O(n)
'''
def enumerateData():
    for i in range(0,size[0]):
        i
        val = df["disease"][i]
        val = val[-1]
        if(i==1):
            print(val)
        
        if(val == 'y'):
            val =0
        else:
            val =int(val)
        df["disease"][i]= val
        
        
if __name__ == "__main__":
    enumerateData()
    
    print(df.head)
    
