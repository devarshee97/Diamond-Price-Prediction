#!/usr/bin/env python
# coding: utf-8

# # 1 Lasso and Ridge regression model for diamond price prediction

# # 1.1 Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ## 1.2 Loading dataset and data preprocessing

# In[2]:


df = pd.read_csv("diamonds.csv")
df.head()


# ## 1.2.1 Exploratory Data Analysis

# In[3]:


df.shape


# The column named "Unnamed: 0" provides no information apart from serial numbers which is present by default; hence this column can be removed

# In[4]:


df.drop("Unnamed: 0", axis=1, inplace=True)


# In[5]:


df.info()


# In[6]:


df.describe()


# From the above summary, we notice that there are certain values of x, y and z that are equal to zero. This indicates that such rows correspond to erroneous data which can be safely removed from our dataset

# In[7]:


cs_1 = df["x"]== 0
cs_2 = df["y"]== 0
cs_3 = df["z"]== 0

idx_val = df[(cs_1) | (cs_2) | (cs_3)].index


# In[8]:


df.drop(idx_val, inplace=True)


# In[9]:


# Check for null values

df.isnull().sum()


# We prepare a separate list of numerical and categorical features which allows us to analyse them more conveniently as and when required

# In[10]:


num_fea = df.select_dtypes(include=["int", "float"]).columns.to_list()
cat_fea = df.select_dtypes(include=["object"]).columns.to_list()

print("Numerical features are ", num_fea)
print("categorical features are ", cat_fea)


# In[11]:


df_num = df[num_fea]
df_cat = df[cat_fea]


# # 1.2.1.1 Data visualization of features

# In[12]:


for i in df_cat.columns:
    plt.figure(figsize=(10,10))
    sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
    plt.xticks(fontsize=14, rotation = "vertical")
    plt.title(i, fontsize=20)
    plt.show()


# In[13]:


corrmat = df_num.corr(method = "spearman")
plt.figure(figsize=(10,10))
g=sns.heatmap(corrmat,annot=True)


# In[14]:


for fea in df_num.columns:
    plt.figure(figsize=(10,10))
    sns.set_style('whitegrid')
    sns.distplot(df[fea], kde=True, bins= 50)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(fea, fontsize=20)
    plt.show()
    print("The skewness is ", skew(df[fea]))
    


# In[15]:


for fea in df_num.columns:
    plt.figure(figsize=(10,8))
    sns.boxplot(df[fea])
    plt.title(fea, fontsize=20)
    plt.show()
    


# # 1.2.2 Feature Engineering

# # 1.2.2.1 Handling ordinal variables
# The categorical features of the datset color, quality and cut are actually nominal variables i.e each of the unique values have an order within them. For e.g, the feature clarity has values "IF" being the highest and "I1" as the lowest. Thus, it is reasonable to expect that the price of a diamond piece should have a strong correlation with the quality of each categorical feature. Hence, we arrange the values of these features as numerical digits with regards to their individual rank.

# In[16]:


cut_dic = {"Ideal": 1, "Premium": 2, "Very Good": 3, "Good":4, "Fair":5}
color_dic = {"D":1, "E":2, "F":3, "G":4, "H":5, "I":6, "J":7}
clarity_dic = {"IF":1, "VVS1":2, "VVS2":3, "VS1":4, "VS2":5, "SI1":6, "SI2":7, "I1":8}

df["cut_order"]= df.cut.map(cut_dic)
df["color_order"]= df.color.map(color_dic)
df["clarity_order"] = df.clarity.map(clarity_dic)


# In[17]:


df.head()


# Upon creating the new columns for the categorical variables, we delete the former as it provides no additional information

# In[18]:


df.drop(["cut", "color", "clarity"], axis=1, inplace=True)


# # 1.2.2.2 Forming new feature by combining dimensions
# The correlation heatmap obtained in the EDA section also reveals a problem with multicollinearity, as a number of variables like x,y and z have strong inter-correlation. This is undesirable especially in a linear regression model, hence, we combine the three dimensions x, y and z to form a new feature as volume of the diamond piece.

# In[19]:


df["Volume"] = df["x"]*df["y"]*df["z"]


# In[20]:


df.drop(["x","y", "z"], axis=1, inplace=True)


# The impact of the feature engineering can be analysed through the correlation heatmap again, which reveals a decrement in the multicollinearity among all features

# In[21]:


corrmat = df.corr(method = "spearman")
plt.figure(figsize=(10,10))
g=sns.heatmap(corrmat,annot=True)


# # 1.2.3 Outlier handling
# 
# As the boxplots in EDA section clearly shows the presence of outliers which can negatively impact our model performance, we treat them using IQR technique.

# In[22]:


q1_crt = df["carat"].quantile(0.25)
q3_crt = df['carat'].quantile(0.75)

iqr_crt = q3_crt - q1_crt

up_crt = q3_crt + 1.5*iqr_crt
low_crt = q1_crt - 1.*iqr_crt

q1_dph = df["depth"].quantile(0.25)
q3_dph = df['depth'].quantile(0.75)

iqr_dph = q3_dph - q1_dph

up_dph = q3_dph + 1.5*iqr_dph
low_dph = q1_dph - 1.*iqr_dph

q1_prc = df["price"].quantile(0.25)
q3_prc = df['price'].quantile(0.75)

iqr_prc = q3_prc - q1_prc

up_prc = q3_prc + 1.5*iqr_prc
low_prc = q1_prc - 1.*iqr_prc

q1_vol = df["Volume"].quantile(0.25)
q3_vol = df['Volume'].quantile(0.75)

iqr_vol = q3_vol - q1_vol

up_vol = q3_vol + 1.5*iqr_vol
low_vol = q1_vol - 1.*iqr_vol

c1 = df['carat']> up_crt
c2 = df['carat']< low_crt
c3 = df['depth']> up_dph
c4 = df['depth']< low_dph
c5 = df['price']> up_prc
c6 = df['price']< low_prc
c7 = df['Volume']> up_vol
c8 = df["Volume"]< low_vol


# In[23]:


df.loc[c1, "carat"] = up_crt
df.loc[c2, "carat"]= low_crt
df.loc[c3, "depth"]= up_dph
df.loc[c4, "depth"]= low_dph
df.loc[c5, "price"]= up_prc
df.loc[c6, 'price']= low_prc
df.loc[c7, 'Volume']= up_vol
df.loc[c8, "Volume"]= low_vol


# # 1.2.3.1 Visulalization to check the impact of outlier treatment

# In[24]:


for fea in df.columns:
    plt.figure(figsize=(10,10))
    sns.set_style('whitegrid')
    sns.distplot(df[fea], kde=True, bins= 50)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(fea, fontsize=20)
    plt.show()
    print("The skewness is ", skew(df[fea]))


# # 1.2.4 Preparation of training and test data

# In[25]:


y = df["price"]
df.drop("price", axis=1, inplace=True)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)


# # 1.2.5 Data scaling

# In[27]:


scaler = StandardScaler()

X_tr_sc = scaler.fit_transform(X_train)
X_tst_sc = scaler.transform(X_test)


# # 1.3 Model implementation

# # 1.3.1 Linear Regression Model(Unregularized)

# In[28]:


lr = LinearRegression()
model_lr = lr.fit(X_tr_sc, y_train)


# In[29]:


tr_scor = model_lr.score(X_tr_sc, y_train)
tst_scor = model_lr.score(X_tst_sc, y_test)

print("Training accuracy ", tr_scor)
print("Test accuracy ", tst_scor)


# # 1.3.2 Ridge Regression Model

# In[30]:


rg = Ridge().fit(X_tr_sc, y_train)
print("Ridge training aacuracy", rg.score(X_tr_sc, y_train))
print("Ridge test accuracy ", rg.score(X_tst_sc, y_test))


# # 1.3.3 Lasso Regression Model

# In[36]:


lsso = Lasso()
model_lsso = lsso.fit(X_tr_sc, y_train)

print("Lasso training aacuracy", model_lsso.score(X_tr_sc, y_train))
print("Lasso test accuracy ", model_lsso.score(X_tst_sc, y_test))


# Predictions of lasso and ridge regression on test dataset

# In[37]:


y_pred_rg = rg.predict(X_tst_sc)
y_pred_lsso = lsso.predict(X_tst_sc)


# # 1.3.4 Evalution of model performance(lasso and ridge)

# In[33]:


print("MAE in Ridge ", mean_absolute_error(y_test, y_pred_rg))
print("MAE in Lasso ", mean_absolute_error(y_test, y_pred_lsso))

print("RMSE in Ridge ", np.sqrt(mean_squared_error(y_test, y_pred_rg)))
print("RMSE in Lasso ", np.sqrt(mean_squared_error(y_test, y_pred_lsso)))

print("R2 score in Ridge ", r2_score(y_test, y_pred_rg))
print("R2 score in Lasso ", r2_score(y_test, y_pred_lsso))


#  # Pickle file creation

# In[38]:


import pickle

file = open('lasso_regression_model.pkl', 'wb')

pickle.dump(lsso, file)


# In[ ]:




