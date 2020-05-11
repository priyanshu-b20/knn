#!/usr/bin/env python
# coding: utf-8

# # KNN Algorithm

# In[1]:


#imports
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,cohen_kappa_score


# In[2]:


iris = load_iris()
features = pd.DataFrame(iris.data,columns=iris.feature_names)
target = pd.DataFrame(iris.target,columns=['target'])
df = pd.concat([features,target],axis=1)
print(df)


# In[3]:


print(iris.DESCR)


# In[4]:


x = iris.data
y = iris.target
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.5,random_state=0)


# In[5]:


corrs = []
for x in features:
    corrs.append(abs(df['target'].corr(df[x])))
plt.figure(figsize=(8,5))    
plt.xticks(np.arange(len(corrs)),features)
plt.ylabel('correlation')
plt.xlabel('features')
plt.bar(np.arange(len(corrs)),corrs, width=0.4)
plt.show()

# In[15]:


#KNN function for one new data point
def knn(x_train,y_train,x_test,k=5):
    score=[]
    for loop in zip(x_train,y_train):
        score.append([distance.euclidean(x_test,loop[0]),loop[1]])
    score.sort(key = lambda x : x[0])
    score = np.array(score)
    return int(stats.mode(score[:k,1])[0])    


# In[16]:


#Implimenting KNN function for whole testing list
pred = []
for x in xtest:
    pred.append(knn(xtrain,ytrain,x,k=5))
pred = np.array(pred)


# In[18]:


#Scores
print('accuracy: ',accuracy_score(ytest,pred))
print('precision: ',precision_score(ytest,pred,average='micro'))
print('recall: ',recall_score(ytest,pred,average='micro'))
print('f1: ',f1_score(ytest,pred,average='micro'))
print('cohen kappa: ',cohen_kappa_score(ytest,pred))


# In[19]:


vals  = np.array([accuracy_score(ytest,pred),precision_score(ytest,pred,average='micro'),recall_score(ytest,pred,average='micro'),f1_score(ytest,pred,average='micro'),cohen_kappa_score(ytest,pred)])*100
plt.figure(figsize=(8,5))    
plt.xticks(np.arange(5),['Accuracy','Precision','Recall','F1','Cohen Kappa'])
plt.ylabel('correlation')
plt.xlabel('features')
plt.bar(np.arange(5),vals, width=0.4)
plt.show()

