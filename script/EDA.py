#!/usr/bin/env python
# coding: utf-8

# ### Monu Babu Baitha

# # Task 1: Data Exploration

# - Load the dataset.
# - Explore the data structure, types, and summary statistics.
# - Visualize relationships between features and the target variable.
# - Identify missing values and outliers.

# In[4]:


get_ipython().system(' pip install pandas')


# In[2]:


get_ipython().system(' pip install numpy')


# In[9]:


get_ipython().system(' pip install seaborn')


# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


get_ipython().system(' pip install keras')


# ##### Load the dataset

# In[17]:


#Loading the dataset
boston = pd.read_csv("BostonHousing.csv")


# In[18]:


boston.head()


# In[19]:


boston.tail()


# In[20]:


print(boston.keys())


# #### Data Preprocessing

# In[22]:


boston.isnull().sum()


# #### Exploring the data structure

# In[24]:


boston.dtypes


# In[25]:


# Summary statistics
print("Summary statistics:")
boston.describe()


# ### Visualize Relationship between features and target variables
# - Here, we consider **medv** as a **target_variable** and remaining as feature variables because medv gives the price. So price will be the main factor to analysis the house
# - we can use different plot for the visualization of the variables relationship 
# 

# In[28]:


# Pairplot 
sns.pairplot(boston)
plt.show()


# In[29]:


sns.pairplot(boston, hue="age")


# In[31]:


# Pairplot between target variable "medv" and feature variable "age"
selected_vars = ['age','medv']
sns.pairplot(boston, vars = selected_vars)
plt.show()


# In[32]:


#Pairplot between target variables "medv" and features variable "crim"
# crim --> crime rate in the town
# medv --> median house value in the town
plt.figure(figsize=(10,6))
sns.pairplot(boston, x_vars=['crim'], y_vars=['medv'], hue='chas')


# In[35]:


plt.figure(figsize=(100,50))
sns.pairplot(boston,x_vars=['crim'],y_vars=['medv'], kind="hist")


# #### Identifying misssing values and outliers

# In[36]:


# Missing values
boston.isnull()


# In[37]:


boston.isnull().sum()


# In[38]:


#As it is clearly shown that there is no any missing values.
# if there is then fill it with 0
boston.fillna(0, inplace=True)


# In[39]:


boston.head


# In[40]:


boston.head()


# In[42]:


# Outliers
sns.boxplot(boston)
plt.show()


# In[44]:


sns.boxplot(boston['medv'])
plt.show()


# In[ ]:




