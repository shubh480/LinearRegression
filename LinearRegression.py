#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import os


# In[101]:


#To find the directory of .csv file


# In[102]:


pwd


# In[103]:


#Using pandas to read the data 
df=pd.read_csv("/home/jovyan/binder/Regression.csv")
df.head


# In[104]:


#Train test splitting procedure
from sklearn.model_selection import train_test_split


# In[105]:


#Inputs and outputs are assigned to X and y
X=df.iloc[:,:-1]
y=df.iloc[:,:4]


# In[106]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)


# In[107]:


from sklearn.linear_model import LinearRegression


# In[108]:


#Procedure of Linear Regression


# In[109]:


regressor = LinearRegression()
regressor.fit(X_train,y_train)

To find predicted values
# In[110]:


regressor.predict(X_train)


# In[111]:


#In order to find Coefficient
print ("Coefficient: ", regressor.coef_)


# In[112]:


#Test score values
print ("Test Score: ", regressor.score(X_test,y_test))


# In[113]:


#Train score values
print ("Train Score: ", regressor.score(X_train,y_train))


# In[114]:


#To find intercept value
print ("Intercept: ", regressor.intercept_)


# In[115]:


import matplotlib.pyplot as plt
import matplotlib.style as style


# In[116]:


style.available
#Using style to plot graph
style.use('fivethirtyeight')
#Plotting marks of Students in training data
plt.scatter(regressor.predict(X_train),regressor.predict(X_train)-(y_train),
           color = "green", s=10, label="Train data")
#Plotting Grades in testing data
plt.scatter(regressor.predict(X_test), regressor.predict(X_test)-y_test,
           color = "blue", s=10, label="Test data")
#plotting line for zero error
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)
#plotting legend
plt.legend(loc="upper right")
#plotting title
plt.title("Students")
#function to show graph
plt.show()


# In[ ]:




