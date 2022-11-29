# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:17:44 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 19:12:42 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 10:43:39 2022

@author: User
"""


import  pandas as pd
from sklearn.model_selection import train_test_split
#import xlsxwriter
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
import pdb
import torch  
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from torch.optim import adam    
from sklearn.model_selection import cross_val_score    
from sklearn.preprocessing import normalize  
from sklearn.utils.validation import column_or_1d  
from csv import writer
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn import metrics
import torch
import warnings
from sklearn.preprocessing import StandardScaler




def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")
def findStatistics(data,y):

        nsamples=data.shape[0]
        nfeatures=data.shape[1]
        
        charac=[]
        
        
        
        classifiers = [
            KNeighborsClassifier(3),
            KNeighborsClassifier(10),
            GaussianNB(),
            SVC(kernel='rbf',max_iter=2000),
            SVC(kernel='sigmoid',max_iter=2000),
            LogisticRegression(max_iter=2000),
            RandomForestClassifier(n_estimators=50,max_depth=7,min_samples_leaf=25,criterion='entropy',bootstrap=True),
            RandomForestClassifier(n_estimators=20,max_depth=35,min_samples_leaf=10,criterion='entropy',bootstrap=True),
            RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_leaf=100,criterion='entropy',bootstrap=True),
            RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=15,criterion='entropy',bootstrap=True),
            RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=70,criterion='entropy',bootstrap=True),
            
            
            
            RandomForestClassifier(n_estimators=50,max_depth=7,min_samples_leaf=25,criterion='entropy',bootstrap=False),
            RandomForestClassifier(n_estimators=20,max_depth=35,min_samples_leaf=10,criterion='entropy',bootstrap=False),
            RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_leaf=100,criterion='entropy',bootstrap=False),
            RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=15,criterion='entropy',bootstrap=False),
            RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=70,criterion='entropy',bootstrap=False),
            
            
            RandomForestClassifier(n_estimators=50,max_depth=7,min_samples_leaf=25,criterion='gini',bootstrap=True),
            RandomForestClassifier(n_estimators=20,max_depth=35,min_samples_leaf=10,criterion='gini',bootstrap=True),
            RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_leaf=100,criterion='gini',bootstrap=True),
            RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=15,criterion='gini',bootstrap=True),
            RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=70,criterion='gini',bootstrap=True),
            

            RandomForestClassifier(n_estimators=50,max_depth=7,min_samples_leaf=25,criterion='gini',bootstrap=False),
            RandomForestClassifier(n_estimators=20,max_depth=35,min_samples_leaf=10,criterion='gini',bootstrap=False),
            RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_leaf=100,criterion='gini',bootstrap=False),
            RandomForestClassifier(n_estimators=150,max_depth=20,min_samples_leaf=15,criterion='gini',bootstrap=False),
            RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=70,criterion='gini',bootstrap=False),
            
                        
      



        ]
        X=np.array(data).astype('float') 
        y = y.ravel()  
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        #scores=[]      
        #for clf in classifiers:
         #   scores.append(cross_validate(clf, data, y, scoring='accuracy',cv=10,error_score="raise",n_jobs=7))

        #for i in range(len(classifiers)):
         # charac.append(scores[i]['test_score'].mean())
        for clf in classifiers:
            clf.fit(X_train,y_train)
            score=clf.score(X_test,y_test)
            charac.append(score)    
        

        return charac


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

def get_data():
    n_samples=int(np.ceil(np.abs(np.random.normal(8000,3000))))
    n_features=np.random.randint(2,150)
    n_informative=np.random.randint(0,n_features)
    n_repeated=np.random.randint(0,n_features-n_informative)
    n_redundant=n_features-n_informative-n_repeated
    n_classes=np.random.randint(2,n_informative//3+1)
    n_clusters_per_class=np.random.randint(1,n_informative//n_classes)
    vec=np.random.uniform(size=n_classes)
    weights=softmax(vec)
    flip_y=np.random.exponential(0.1)
    class_sep=np.random.uniform(0.1,5)
    data = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, 
                            n_classes=n_classes, weights=weights, flip_y=flip_y, class_sep=class_sep, 
                            hypercube=True, shift=2, scale=1, shuffle=True, random_state=None)
    scales=np.random.uniform(0.1,1000,size=n_features)
    shifts=np.random.uniform(0.1,1000,size=n_features)
    for i in range(n_features):
        data[0][:,i]=data[0][:,i]*scales[i]+shifts[i]
    return data    

    
def makeHyper():
  max_depth=np.random.choice(np.array(list(range(50))))
  min_samples_leaf=np.random.choice(np.array(list(range(2,50,1))))
  criterion=np.random.choice(["gini", "entropy"])
  n_estimators=np.random.choice(np.array(list(range(2,100,5))))
  bootstrap=np.random.choice([0,1])
  criterionNum=1 if criterion=="gini" else 2
  return max_depth,min_samples_leaf,n_estimators,bootstrap,criterion,criterionNum

batch_size=50

def make_batch():
    batch=[]
    real_scores=[]
    i=0
    while(i<25):
      try:
            i+=1
            print(i)
                     
            X,y=get_data()
            scaler=StandardScaler()
            X=scaler.fit_transform(X)
            xpd=pd.DataFrame(X)
            ypd=pd.Series(y)
            statistics=np.array(findStatistics(xpd,ypd))
            max_depth,min_samples_leaf,n_estimators,bootstrap,criterion,criterionNum=makeHyper()
            hyperTensor=np.array([max_depth,min_samples_leaf,n_estimators,bootstrap,criterionNum])
            combtensor=np.concatenate((statistics,hyperTensor))
            
            bootstrapa=True if bootstrap==1 else False
            #make labels
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
            model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,bootstrap=bootstrapa,criterion=criterion)
            model.fit(X_train,y_train)
            real_score= model.score(X_test,y_test)
            batch.append(combtensor)
            real_scores.append(real_score)

     
   
                                
      except KeyboardInterrupt:
          print("breakkkkkkkkkkk")
          break
        
      except:
          continue  
    return batch,real_scores  