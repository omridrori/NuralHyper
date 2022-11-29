# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:26:10 2022

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
    n_samples=int(np.ceil(np.abs(np.random.normal(3000,1500))))
    n_features=np.random.randint(0,120)
    n_informative=np.random.randint(0,n_features)
    n_repeated=np.random.randint(0,n_features)
    n_redundant=n_features-n_informative-n_repeated
    n_classes=np.random.randint(2,n_informative)
    n_clusters_per_class=np.random.randint(1,n_informative)
    vec=np.random.uniform(size=n_classes)
    weights=softmax(vec)
    perc_flip=np.random.choice([0.1,0.2])
    flip_y=np.random.exponential(perc_flip)
    
    
    min_sep=np.random.choice([0.01,0.1,0.5,1])
    max_sep=np.random.randint(2,10)

    class_sep=np.random.uniform(0.1,5)
    
    data = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, 
                            n_classes=n_classes, weights=weights, flip_y=flip_y, class_sep=class_sep, 
                            hypercube=True, shuffle=True, random_state=None)
    #scales=np.random.uniform(0.1,1000,size=n_features)
    #shifts=np.random.uniform(0.1,1000,size=n_features)
    #for i in range(n_features):
     #   data[0][:,i]=data[0][:,i]*scales[i]+shifts[i]
    return data    




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(31,40)
        self.fc2=nn.Linear(40,64)
        self.fc6=nn.Linear(64,128)
        self.fc7=nn.Linear(128,256)
        self.fc8=nn.Linear(256,128)
        self.fc3=nn.Linear(128,32)
        self.fc4=nn.Linear(32,8)
        self.fc5=nn.Linear(8,1)


    def forward(self,x):
        
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc3(x))
        
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
    
     
     
    
    
net = Net()
net.load_state_dict(torch.load(r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\rf1'))


def makeHyper():
  max_depth=np.random.choice(np.array(list(range(50))))
  min_samples_leaf=np.random.choice(np.array(list(range(2,50,1))))
  criterion=np.random.choice(["gini", "entropy"])
  n_estimators=np.random.choice(np.array(list(range(0,100,10))))
  if n_estimators==0:
      n_estimators=1
  bootstrap=np.random.choice([0,1])
  criterionNum=1 if criterion=="gini" else 2
  return max_depth,min_samples_leaf,n_estimators,bootstrap,criterion,criterionNum

optimizer = optim.Adam(net.parameters(), lr=3e-4)

los=nn.L1Loss()







epochreal=1
SumLoss=0
for epoch in range(1,100000000):

    print('************************\n')
    
    
    optimizer.zero_grad()
        #create summary of all the data in embedding space
    try:
        if(epoch%100==0):
            torch.save(net.state_dict(), r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\rf3')


        X,y=get_data()
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        xpd=pd.DataFrame(X)
        ypd=pd.Series(y)
        statistics=findStatistics(xpd,ypd)
        statistics=torch.tensor(statistics).float()


        print('X shape: {}\n'.format(X.shape))
              ################find real score
        max_depth,min_samples_leaf,n_estimators,bootstrap,criterion,criterionNum=makeHyper()
        hyperTensor=torch.tensor([max_depth,min_samples_leaf,n_estimators,bootstrap,criterionNum]).float()
        print("hyper parameters: {}\n".format(hyperTensor))
        bootstrapa=True if bootstrap==1 else False
        model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,bootstrap=bootstrapa,criterion=criterion)
        #scores=cross_val_score(model, X, y, cv=15,n_jobs=7)
        #real_score=torch.tensor(scores.mean()).float()
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        model.fit(X_train,y_train)
        real_score=torch.tensor([model.score(X_test,y_test)]).float()
        ## real_score=torch.tensor([model.score(X_test,y_test)]).float()

        print('real score: {}\n'.format(real_score))
        combtensor=torch.cat((statistics,hyperTensor))

        score_predicted=net(combtensor)
        if(torch.isnan(score_predicted).item()):
            continue
        print('predicted score: {}\n'.format(score_predicted))


        loss = los(score_predicted, real_score)
        loss.backward()
        optimizer.step()
              # print statistics
        SumLoss+=loss.item()
        print('epoch number: {},the current loss is {}\n'.format(epochreal,loss.data))
        print('mean loss is: {}'.format(SumLoss/epochreal))
        epochreal+=1


         
    except KeyboardInterrupt:
        print("breakkkkkkkkkkk")
        break

    except:
        continue  
    


    




    



    
    
    
    
    