# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:24:11 2022

@author: luisa
"""
from sklearn.model_selection import train_test_split as split

import pandas as pd
import os

from AttributeManager import AttributeManager
from Experiments import Experiment

pathD = '../data/'  # Data's path
contenido = os.listdir(pathD)
csv = []
for fichero in contenido:
    if os.path.isfile(os.path.join(pathD, fichero)) and fichero.endswith('.csv'):
        index = fichero.index('.')
        fichero = fichero[:index]
        csv.append(fichero)
    
N = 30 # Iterations
    
nBest = 5 # Number of best attributes
        
dPolynomial = 5 # Dgree Polynomial

pathSave = 'results/'

for fileName in csv:
    print(fileName)
    
    data = pd.read_csv(pathD + fileName + '.csv')
    
    X = data.iloc[:, :-1]
    
    # Select  only numeric values
    X.dropna(inplace=True)
    idx = X.dtypes.values != 'object'
    X = X.iloc[:, idx]
    
    Y = data.iloc[:, -1]
    
    # List of each experiment
    f1_E1 = []
    prec_E1 = []
    rec_E1 = []
    
    f1_E2 = []
    prec_E2 = []
    rec_E2 = []
    
    f1_E3 = []
    prec_E3 = []
    rec_E3 = []
    
    exp = Experiment()
    
    for i in range(N):
    
        X_train, X_test, y_train, y_test = split(X, Y, train_size=0.2)
        
    	# Experiment I
        f1, prec, rec = exp.computeClassifiers(X_train, X_test, y_train, y_test)
        f1_E1.append(f1)
        prec_E1.append(prec)
        rec_E1.append(rec)
        
        # Experiment II
        am = AttributeManager()
        # Applicar MrMR
        listAttr = am.getBestMRMR(X_train, y_train, nBest)# Number of atributes
        X_tr = X_train[listAttr]
        X_ts = X_test[listAttr]
        
        f1, prec, rec = exp.computeClassifiers(X_tr, X_ts, y_train, y_test)
        f1_E2.append(f1)
        prec_E2.append(prec)
        rec_E2.append(rec)
        
        # Experiment III
        com_tr = am.computeCombinations(listAttr, dPolynomial) # Dgree polynomial
        com_ts = am.computeCombinations(listAttr, dPolynomial)
        
        Xf_tr, nNames_tr, namesC_tr = am.computeAtts(X_tr, com_tr)
        Xf_ts, nNames_ts, namesC_ts = am.computeAtts(X_ts, com_ts)
        
        # Select best 
        listAttr_new = am.getBestMRMR(Xf_tr, y_train, nBest)
        
        # New Data of MrMR
        Xfm_tr = Xf_tr[listAttr_new]
        Xfm_ts = Xf_ts[listAttr_new]
        
        f1, prec, rec = exp.computeClassifiers(Xfm_tr, Xfm_ts, y_train, y_test)
        f1_E3.append(f1)
        prec_E3.append(prec)
        rec_E3.append(rec)
    
    print('Creando archivos')
    #Create DataFrame and export to csv
    exp.exportDataCSV(str(1) + '_' + str(nBest) + str(dPolynomial), fileName, pathSave, f1_E1, prec_E1, rec_E1)
    exp.exportDataCSV(str(2) + '_' + str(nBest) + str(dPolynomial), fileName, pathSave, f1_E2, prec_E2, rec_E2)
    exp.exportDataCSV(str(3) + '_' + str(nBest) + str(dPolynomial), fileName, pathSave, f1_E3, prec_E3, rec_E3)
    print('Archivos creados\n')