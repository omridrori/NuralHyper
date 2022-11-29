# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 19:17:22 2022

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
from rfutil import findStatistics
from rfutil import get_data
from rfutil import make_batch
from rfutil import makeHyper



def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")






def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()



    

batch_size=50




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
net.load_state_dict(torch.load(r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\rf5'))



      
 
optimizer = optim.Adam(net.parameters(), lr=0.000001)

los=nn.L1Loss()


 
SumLoss=0.021123416599009064*1092
for epoch in range(1092,100000000):

    print('************************\n')
    
    
    optimizer.zero_grad()
        #create summary of all the data in embedding space
    if(epoch%10==0):
        torch.save(net.state_dict(),  r'C:\Users\User\Desktop\מדמח\מודלימז\פרויקט חדש\rf6')


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
    real_scores_numpy=[round(float(real_scores_numpy[k]),3) for k in range(len(real_scores_numpy))]
    score_predicted_numpy=[round(float(score_predicted_numpy[k]),3) for k in range(len(real_scores_numpy))]
    ziped=list(zip(real_scores_numpy,score_predicted_numpy))
    #print('real score: {}\n'.format(real_scores))
    print(ziped)
    #print('predicted score: {}\n'.format(score_predicted_numpy.reshape(-1)))

    print(score_predicted.shape)
    print(real_scores.shape)
    loss = los(score_predicted, real_scores).float()
    print(2)
    loss.backward()
    optimizer.step()
          # print statistics
    SumLoss+=loss.item()
    print('---------------------------')
    print('epoch number: {},the current loss is {}\n'.format(epoch,loss.data))
    print('mean loss is: {}'.format(SumLoss/epoch))


         

    



                 
    
