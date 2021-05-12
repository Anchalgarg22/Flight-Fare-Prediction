#!/usr/bin/env python
# coding: utf-8

# ## Flight Fare Prediction

# ### Description

# Guessing the flight prices can be very hard sometimes, today we might see a price but when we check out the same flight the price might be different. We might have often heard travelers saying that flight ticket prices are so unpredictable. Here we will be provided with different prices of flight tickets for various airlines between the months of March and June of 2019 and between various cities.

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading training dataset
traindf = pd.read_excel('C:\Praxis\Dataset\Flight Fare Prediction\Data_Train.xlsx')
traindf.head(2)


# In[3]:


#loading test dataset
testdf = pd.read_excel('C:\Praxis\Dataset\Flight Fare Prediction\Test_set.xlsx')
testdf.head(2)


# ### Feature Engineering

# In[4]:


traindf.info()


# In[5]:


traindf.describe()


# In[6]:


traindf.describe(include = 'all')


# In[7]:


traindf.columns #calling columns of training dataset


# In[8]:


traindf.isnull().sum() #There are only 2 null values in our dataset, one in Route and one in Total_Stops


# In[9]:


traindf.dropna(inplace = True) #dropping null values from training dataset


# In[10]:


traindf[traindf.duplicated()].head() #checking duplicate values in our training dataframe.


# In[11]:


traindf.drop_duplicates(keep='first',inplace= True) #droppping duplicate values from training dataset


# In[12]:


traindf.shape


# In[13]:


traindf['Airline'].unique() #unique airlines


# In[14]:


traindf['Airline'].value_counts()  #mostly used airline is Jet Airways and Indigo


# In[15]:


traindf['Route'].value_counts() #often traveled route is Delhi -> Bombay -> Cochin after that Bengaluru -> Delhi


# In[16]:


testdf.isnull().sum() #no null values in test dataset


# In[17]:


traindf.head()


# In[18]:


# Dividing data into features and labels
# converting whole duration of journey into minutes for both train and test dataset

traindf['Duration'] = traindf['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
testdf['Duration'] = testdf['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[19]:


# separating Date_of_Journey into 2 variables journey day and journey month for train set

traindf["Journey_day"] = traindf['Date_of_Journey'].str.split('/').str[0].astype(int)
traindf["Journey_month"] = traindf['Date_of_Journey'].str.split('/').str[1].astype(int)
traindf.drop(["Date_of_Journey"], axis = 1, inplace = True)

# similarily Dep_Time into departure hour and departure minute for train set

traindf["Dep_hour"] = pd.to_datetime(traindf["Dep_Time"]).dt.hour
traindf["Dep_min"] = pd.to_datetime(traindf["Dep_Time"]).dt.minute
traindf.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time into arrival hour and arrival minute for train set

traindf["Arrival_hour"] = pd.to_datetime(traindf.Arrival_Time).dt.hour
traindf["Arrival_min"] = pd.to_datetime(traindf.Arrival_Time).dt.minute
traindf.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[20]:


# separating Date_of_Journey into 2 variables journey day and journey month for test set

testdf["Journey_day"] = testdf['Date_of_Journey'].str.split('/').str[0].astype(int)
testdf["Journey_month"] = testdf['Date_of_Journey'].str.split('/').str[1].astype(int)
testdf.drop(["Date_of_Journey"], axis = 1, inplace = True)

#  similarily Dep_Time into departure hour and departure minute for test set

testdf["Dep_hour"] = pd.to_datetime(testdf["Dep_Time"]).dt.hour
testdf["Dep_min"] = pd.to_datetime(testdf["Dep_Time"]).dt.minute
testdf.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time into arrival hour and arrival minute for test set

testdf["Arrival_hour"] = pd.to_datetime(testdf.Arrival_Time).dt.hour
testdf["Arrival_min"] = pd.to_datetime(testdf.Arrival_Time).dt.minute
testdf.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[21]:


traindf.info()


# ### Data Visualization

# In[22]:


# Plotting Price vs Airline plot
sns.catplot(y = "Price", x = "Airline", data = traindf.sort_values("Price", ascending = False), kind="strip", height = 5, aspect = 4)
plt.show()


# In[23]:


# Plotting Price vs Source
sns.catplot(y = "Price", x = "Source", data = traindf.sort_values("Price", ascending = False), kind="violin", height = 4, aspect = 3)
plt.show()


# In[24]:


# Plotting Box plot for Price vs Destination
sns.catplot(y = "Price", x = "Destination", data = traindf.sort_values("Price", ascending = False), kind="box", height = 4, aspect = 3)
plt.show()


# In[25]:


# Plotting Bar chart for Months (Duration) vs Number of Flights

plt.figure(figsize = (10, 5))
plt.title('Count of flights month wise')
ax=sns.countplot(x = 'Journey_month', data = traindf)
plt.xlabel('Month')
plt.ylabel('Count of flights')
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom', color= 'black')


# In[26]:


# Plotting Ticket Prices VS Airlines
plt.figure(figsize = (15,4))
plt.title('Price VS Airlines')
plt.scatter(traindf['Airline'], traindf['Price'])
plt.xticks
plt.xlabel('Airline')
plt.ylabel('Price of ticket')
plt.xticks(rotation = 90)


# In[27]:


# Plotting Bar chart for Types of Airline vs Number of Flights
plt.figure(figsize = (20,5))
plt.title('Count of flights with different Airlines')
ax = sns.countplot(x = 'Airline', data =traindf)
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 45)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom', color= 'black')


# ### Correlation between all Features

# In[28]:


# Plotting Correation

plt.figure(figsize = (15,15))
sns.heatmap(traindf.corr(), annot = True, cmap = "RdYlGn")
plt.show()


# #### Variables are not correlated with each other. Hence, we can start building our model.

# #### Defining X and Y training and test set

# In[29]:


#changing variable type to integer for training dataset

traindf[['Journey_day','Journey_month','Duration','Price']]=traindf[['Journey_day','Journey_month','Duration','Price']].astype(int)


# In[30]:


X = traindf[['Journey_day','Journey_month','Duration']]


# In[31]:


y = traindf[['Price']]


# In[32]:


print("The size of training input is", X.shape)
print("The size of training output is", y.shape)


# In[33]:


from sklearn.model_selection import train_test_split #importing train test split


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[35]:


print("The size of training input is", X_train.shape)
print("The size of training output is", y_train.shape)
print("The size of test input is", X_test.shape)
print("The size of test output is", y_test.shape)


# ### Decision Tree Regressor

# In[36]:


#importing required libraries
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
from sklearn.metrics import r2_score

#defining our model
tree = DecisionTreeRegressor(max_depth = 4, random_state = 5)


# In[37]:


#fitting our model
tree.fit(X_train,y_train)


# In[38]:


#predicting from training data
y_train_pred = tree.predict(X_train)


# In[39]:


# Calculating Mean Absolute Percentage Error

from sklearn.metrics import mean_squared_error as mse

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[40]:


#Printing our results

print("Train Results for Decision Tree Regressor Model:")
print("Root Mean squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))


# In[41]:


#predicting test dataset

y_test_pred = tree.predict(X_test)


# In[42]:


#printing our results for training dataset

print("Test Results for Decision Tree Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test.values, y_test_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_test.values, y_test_pred)))
print("R-Squared: ", r2_score(y_test.values, y_test_pred))


# ### Random Forest Regression

# In[43]:


#importing Required Libraries
from sklearn.ensemble import RandomForestRegressor

#defining our model
forest = RandomForestRegressor(max_depth = 4, random_state = 5)


# In[44]:


#fitting our model
forest.fit(X_train,y_train)


# In[45]:


#predicting train dataset

y_train_pred = forest.predict(X_train)


# In[46]:


#printing our result for training dataset

print("Train Results for Random Forest Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))


# In[47]:


#predicting test dataset

y_test_pred = forest.predict(X_test)


# In[48]:


print("Test Results for Random Forest Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# ### Ridge Regression

# In[49]:


#importing required library
from sklearn.linear_model import Ridge

#defining our model
ridge = Ridge(random_state = 5)


# In[50]:


#fitting our model
ridge.fit(X_train,y_train)


# In[51]:


#predicting train dataset

y_train_pred = ridge.predict(X_train)


# In[52]:


#printing our result for training dataset

print("Train Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train.values, y_train_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_train.values, y_train_pred)))
print("R-Squared: ", r2_score(y_train.values, y_train_pred))


# In[53]:


#predicting test dataset

y_test_pred = ridge.predict(X_test)


# In[54]:


print("Test Results for Ridge Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# ### Lasso Regression

# In[55]:


#importing required library
from sklearn.linear_model import Lasso

#defining our model
lasso = Lasso(random_state = 5)


# In[56]:


#fitting our model
lasso.fit(X_train,y_train)


# In[57]:


#predicting train dataset

y_train_pred = lasso.predict(X_train)


# In[58]:


#printing our result for training dataset

print("Train Results for Lasso Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_train, y_train_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_train, y_train_pred)))
print("R-Squared: ", r2_score(y_train, y_train_pred))


# In[59]:


#predicting test dataset

y_test_pred = lasso.predict(X_test)


# In[60]:


print("Test Results for Lasso Regressor Model:")
print("Root Mean Squared Error: ", sqrt(mse(y_test, y_test_pred)))
print("Mean Absolute % Error: ", round(mean_absolute_percentage_error(y_test, y_test_pred)))
print("R-Squared: ", r2_score(y_test, y_test_pred))


# ### Comparing our Model

# In[61]:


from prettytable import PrettyTable


# In[62]:


# Training = Tr.
# Testing = Te.
x = PrettyTable()
x.field_names = ["Model Name", "Tr. RMSE", "Tr. MA%E", "Tr. R-Squared", "Te. RMSE", "Te. MA%E", "Te. R-Squared",]
x.add_row(['Decision Tree Regressor','2974.88','65','0.59','2919.47','64','0.58'])
x.add_row([" ------------------------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "])
x.add_row(['Random Forest Regressor','2918.4500','65','0.60','2919.48','64','0.58'])
x.add_row([" ------------------------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "])
x.add_row(['Ridge Regression','3955.33','38','0.28','3798.84','38','0.29'])
x.add_row([" ------------------------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "," ----------- "])
x.add_row(['Lasso Regression','3955.33','59','0.28','3798.84','59','0.29'])
print(x)


# ### Final Model

# As we can see in above table, Ridge Regression is performing better among all. So, we will choose Ridge Regression as Final Model and will create pickle file for same.

# ### Creating pickle file for our model

# In[63]:


import pickle
pickle.dump(ridge, open('flightprediction.pkl','wb'))


# In[ ]:




