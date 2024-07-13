#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# # Loading Data

# In[3]:


input_test = pd.read_csv("test.csv")
input_test


# In[4]:


input_test.info()


# In[5]:


train_data = pd.read_csv("train.csv")
train_data.info()
train_data


# In[6]:


output_test = pd.read_csv("gender_submission.csv")
output_test


# # Checking Coorrelation

# In[7]:


sns.heatmap(train_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]].corr(), annot= True)


# # Sibsb - Number of Siblings

# In[8]:


sns.barplot(x = "SibSp", y = "Survived", data = train_data)
# Those with high siblings value hass less chance to survive


# # Age Parameter

# In[9]:


age_visuals = sns.FacetGrid(train_data, col = "Survived").map(sns.histplot, "Age")
#less age people have more chances of survival


# # Gender Parameter

# In[10]:


sns.barplot(x = "Sex", y = "Survived", data = train_data)
# females have more chances


# # Pclass Parameter

# In[11]:


sns.barplot(x = "Pclass", y = "Survived", data =  train_data, hue = "Sex")
# 1st class means rich people have more chances


# # Embarked

# In[12]:


# There are some null values so i have to fill these null values with something


# In[13]:


train_data["Embarked"].value_counts()


# In[14]:


# Most of them are S, so i will fill these two null values with S
train_data["Embarked"] = train_data["Embarked"].fillna("S")


# In[15]:


sns.barplot(x = "Embarked", y = "Survived", data = train_data)


# In[16]:


# Here people from C Ship stop they are rich and mostly are female thus they have some effect on survival rate otherwise they are of no use


# # Preparing The ML Model

# In[17]:


# First fix all the null values


# In[18]:


train_data.info()


# In[19]:


mean = train_data["Age"].mean()
std = train_data["Age"].std()
no_of_null = train_data["Age"].isnull().sum()


# In[20]:


print(mean, std, no_of_null)


# In[21]:


random_age = np.random.randint(mean-std, mean+std, size = no_of_null)


# In[22]:


random_age


# In[23]:


mean_test = input_test["Age"].mean()
std_test = input_test["Age"].std()
nullval = input_test["Age"].isnull().sum()
random_age_test = np.random.randint(mean_test-std_test, mean_test+std_test, nullval)


# In[24]:


a = train_data["Age"].copy()
a[np.isnan(a)] = random_age
train_data["Age"] = a
train_data["Age"]


# In[25]:


a = input_test["Age"].copy()
a[np.isnan(a)] = random_age_test
input_test["Age"] = a
input_test


# In[26]:


train_data = train_data.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis = 1)
train_data
input_test = input_test.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis = 1)


# In[27]:


gender = {"male":0, "female":1}
train_data["Sex"] = train_data["Sex"].map(gender)
train_data["Sex"]
input_test["Sex"] = input_test["Sex"].map(gender)


# In[28]:


ports = {"S": 0, "C": 1, "Q": 2}
train_data["Embarked"] = train_data["Embarked"].map(ports)
train_data["Embarked"]
input_test["Embarked"] = input_test["Embarked"].map(ports)
train_data.info()


# In[29]:


input_test["Fare"] = input_test["Fare"].fillna(35.627188489208635)
input_test.info()


# # Spillting the data

# In[30]:


x = train_data.drop(["Survived"], axis = 1)
y = train_data["Survived"]
x_test = input_test


# # Scalling the data

# In[45]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x)
x_test = ss.fit_transform(x_test)
y_test = output_test.drop(["PassengerId"], axis = 1)


# In[46]:


y_test


# # Classification

# In[47]:


logistic_classifier = LogisticRegression()
svc_classifier = SVC()
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier()
rf_classifier = RandomForestClassifier()


# In[48]:


logistic_classifier.fit(x_train, y)
svc_classifier.fit(x_train, y)
dt_classifier.fit(x_train, y)
knn_classifier.fit(x_train, y)
rf_classifier.fit(x_train, y)


# In[49]:


log_pred = logistic_classifier.predict(x_test)
svc_pred = svc_classifier.predict(x_test)
dt_pred = dt_classifier.predict(x_test)
knn_pred = knn_classifier.predict(x_test)
rf_pred = rf_classifier.predict(x_test)


# In[50]:


from sklearn.metrics import accuracy_score
log_acc = accuracy_score(y_test, log_pred)
svc_acc = accuracy_score(svc_pred, y_test)
dt_acc = accuracy_score(dt_pred, y_test)
knn_acc = accuracy_score(knn_pred, y_test)
rf_acc = accuracy_score(rf_pred, y_test)


# In[51]:


print(log_acc)
print(svc_acc)
print(dt_acc)
print(knn_acc)
print(rf_acc)


# In[ ]:




