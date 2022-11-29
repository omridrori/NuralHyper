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
torch.set_warn_always(False)
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
            DecisionTreeClassifier(max_depth=7,min_samples_leaf=25,criterion='entropy'),
            DecisionTreeClassifier(max_depth=35,min_samples_leaf=10,criterion='entropy'),
            DecisionTreeClassifier(max_depth=2,min_samples_leaf=100,criterion='entropy'),
            
            DecisionTreeClassifier(max_depth=20,min_samples_leaf=5,criterion='gini'),
            DecisionTreeClassifier(max_depth=2,min_samples_leaf=5,criterion='gini'),
            DecisionTreeClassifier(max_depth=20,min_samples_leaf=100,criterion='gini'),



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

def make_data():
    n_samples=int(np.ceil(np.abs(np.random.normal(15000,5000))))
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
  criterionNum=1 if criterion=="gini" else 2
  return max_depth,min_samples_leaf,criterion,criterionNum

batch_size=50

def make_batch():
    batch=[]
    real_scores=[]
    i=0
    while(i<25):
      try:
                     
            X,y=make_data()
            scaler=StandardScaler()
            X=scaler.fit_transform(X)
            xpd=pd.DataFrame(X)
            ypd=pd.Series(y)
            statistics=np.array(findStatistics(xpd,ypd))
            max_depth,min_samples_leaf,criterion,criterionNum=makeHyper()
            hyperTensor=np.array([max_depth,min_samples_leaf,criterionNum])
            combtensor=np.concatenate((statistics,hyperTensor))
            
            
            #make labels
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
            model=DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf,criterion=criterion)
            model.fit(X_train,y_train)
            real_score= model.score(X_test,y_test)
            batch.append(combtensor)
            real_scores.append(real_score)
            i+=1
            print(i)
     
   
                                
      except KeyboardInterrupt:
          print("breakkkkkkkkkkk")
          break
        
      except:
          continue  
    return batch,real_scores  


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15,32)
        self.fc2=nn.Linear(32,64)
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
net.load_state_dict(torch.load(r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\batches117'))



      
 
optimizer = optim.Adam(net.parameters(), lr=0.00000001)

los=nn.L1Loss()

 
SumLoss=0.021123416599009064*1092
for epoch in range(1092,100000000):

    print('************************\n')
    
    
    optimizer.zero_grad()
        #create summary of all the data in embedding space
    if(epoch%100==0):
        torch.save(net.state_dict(), r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\batches117')
        with open(r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\event5.csv', 'a') as f_object:
            writer_object = writer(f_object)
            listd=[epoch,SumLoss/epoch]
            writer_object.writerow(listd)
            f_object.close()

    batch,real_scores=make_batch()
    real_scores_numpy=real_scores
    batch=torch.tensor(batch).float()
    real_scores=torch.tensor(real_scores)
    score_predicted=net(batch)
    score_predicted=score_predicted.view(-1)
    score_predicted_numpy=score_predicted.detach().numpy()
    array_sum = np.sum(score_predicted_numpy)
    array_has_nan = np.isnan(array_sum)
    
    if(np.isnan(array_sum)):
        continue
    score_predicted_numpy=score_predicted_numpy.reshape(-1)
    real_scores_numpy=[round(real_scores_numpy[k],3) for k in range(len(real_scores_numpy))]
    score_predicted_numpy=[round(score_predicted_numpy[k],3) for k in range(len(real_scores_numpy))]
    ziped=list(zip(real_scores_numpy,score_predicted_numpy))
    #print('real score: {}\n'.format(real_scores))
    print(ziped)
    #print('predicted score: {}\n'.format(score_predicted_numpy.reshape(-1)))

    print(score_predicted.shape)
    print(real_scores.shape)
    loss = los(score_predicted, real_scores)
    loss.backward()
    optimizer.step()
          # print statistics
    SumLoss+=loss.item()
    print('---------------------------')
    print('epoch number: {},the current loss is {}\n'.format(epoch,loss.data))
    print('mean loss is: {}'.format(SumLoss/epoch))


         

    



                 
    
