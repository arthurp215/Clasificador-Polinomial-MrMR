# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:23:08 2022

@author: luisa
"""

from mrmr import mrmr_classif
import itertools
import numpy as np
import pandas as pd

class AttributeManager: 
    
    def __init__(self):
        pass
    
    def getBestMRMR(self, X, Y, idx=0, allDicc = False):
        row, column = X.shape
        #classification = [] # Create dicc
        # De best attributes 
        #for i in range(column, 0, -1):
        classification = mrmr_classif(X, Y, column)
        # The best elements to index
        if allDicc:
            return classification
        else:
            return classification[:idx]
    
    def computeCombinations(self, attNames, degree):
        atts = list(range(len(attNames)))
        return  itertools.combinations_with_replacement(atts, degree)
    
    def computeAtt(self, X=None, combinacion=None, asSeries=True):
        if X is None:
            return None
        elif type(X) is pd.DataFrame:
            X = X.copy()
            X = X.values
        elif type(X) is np.ndarray:
            X = X
        att = X[:, combinacion[0]] # Empieza con el primer elemento
        for i in range(1, len(combinacion)):
            att  = np.multiply(att, X[:, combinacion[i]])
        if  asSeries:
            att = pd.Series(att, str(combinacion))
        return att
    
    def computeAtts(self,  X=None, combinaciones=None, asDataFrame=True):
        if X is None or combinaciones is None:
            return None, None, None
        if type(X) is pd.DataFrame:
            names = list(X.columns)
            X_df = X.copy()
            X_a = X_df.values
        elif type(X) is np.ndarray:
            names = [str(i) for i in range(0, X_a.shape[1])]            
        names = names.copy()
        newNames = []
        X_new  = None    
        #np.reshape(X_new, (X_new.shape[0], 1))
        for comb in combinaciones:
            
            newAtt = self.computeAtt(X_a, comb, asSeries=False)
            newAtt_l = newAtt.tolist()
            X_df[str(comb)] = newAtt_l
            '''
            if X_new is None:
                X_new = newAtt
                X_new = np.reshape(X_new, (X_new.shape[0], 1))
            else:
                X_new = np.hstack((X_new, np.reshape(newAtt, (X_new.shape[0], 1))))
                #X_new = np.reshape(X_new, (X_new.shape[0], len(newNames) + 1))'''
            names.append(str(comb))
            newNames.append(comb)# Nombre atributo
        #X_new = np.hstack((X, X_new))    
        if   asDataFrame:
            return X_df, names, newNames
        else:
            return X_df.values, names, newNames
    
