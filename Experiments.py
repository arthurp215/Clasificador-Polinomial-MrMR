
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np

class Experiment:
	
    def __init__(self):
        pass

    def selectClassifier(self, X_tr, Y_tr, clf):
        classifier = 0
        if clf == 1:
            classifier = LogisticRegression(max_iter=1000000) 
        elif clf == 2:
            classifier = KNeighborsClassifier(n_neighbors = 7)
        elif clf == 3:
            classifier = DecisionTreeClassifier()
        elif clf == 4:
            classifier = SVC(decision_function_shape='ovo')
        elif clf == 5:
            classifier = GaussianNB()
        elif clf == 6:
            classifier = MLPClassifier(random_state=1, max_iter=1000000)
        else:
            print('Esta opción no existe')
        
        self.clfr = classifier.fit(X_tr, Y_tr)

    def test(self, X_test, y_test):
        y_pred = self.clfr.predict(X_test)	
        # Performance
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        return f1, prec, rec

    def computeClassifiers(self, X_tr, X_ts, y_tr, y_ts):
        f1_l = []
        prec_l = []
        rec_l = []
        cont = 1
        # Entrenar clasificador
        while(cont < 7):
            self.selectClassifier(X_tr, y_tr, cont)
            f1, prec, rec = self.test(X_ts, y_ts)
            f1_l.append(f1)
            prec_l.append(prec)
            rec_l.append(rec)
            cont += 1
    
        return f1_l, prec_l, rec_l
    
    def exportDataCSV(self, nExp, fileName, pathSave, f1, prec, rec):
        column = ['lrF','knnF','dtF','svcF','nbF','mlpcF','lrP','knnP','dtP','svcP','nbP','mlpcP','lrR','knnR','dtR','svcR','nbR','mlpcR']
        
        matrix = np.hstack((f1,prec,rec)) # Stack
        df = pd.DataFrame(matrix, columns=column) # Create Df
        men = df.mean().to_dict() # Mean
        std = df.std().to_dict() # Std
        dfFinal = df.append([men,std], ignore_index=True) # Add mean and std to Df
        dfFinal.to_csv(pathSave + fileName + '_experiment' + nExp + '.csv') # Export df
        
        
