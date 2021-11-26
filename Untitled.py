#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Import libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjusts the configuration of the plots we will create


# Read in the data

df = pd.read_csv(r'file:///Users/joshuabuchanan/Downloads/movies.csv')


# In[8]:


df.head()


# In[11]:


# Seeing if there's any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing*100))


# In[10]:


# Data types for our columns

df.dtypes


# In[12]:


# Are there any Outliers?

df.boxplot(column=['gross'])


# In[13]:


df.drop_duplicates()


# In[14]:


# Order our Data a little bit to see

df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[15]:


sns.regplot(x="gross", y="budget", data=df)


# In[16]:


sns.regplot(x="score", y="gross", data=df)


# In[17]:


# Correlation Matrix between all numeric columns

df.corr(method ='pearson')


# In[18]:


df.corr(method ='kendall')


# In[19]:


df.corr(method ='spearman')


# In[20]:


correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[21]:


# Using factorize - this assigns a random numeric value for each unique categorical value

df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[22]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[23]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[24]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# In[25]:


# We can now take a look at the ones that have a high correlation (> 0.5)

strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


# In[26]:


# Looking at the top 15 compaies by gross revenue

CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[27]:


df['Year'] = df['released'].astype(str).str[:4]
df


# In[28]:


df.groupby(['company', 'year'])[["gross"]].sum()


# In[29]:


CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[30]:


CompanyGrossSum = df.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[31]:


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[32]:


df_numerized = df


for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[33]:


df_numerized.corr(method='pearson')


# In[34]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[ ]:


for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name]= df[col_name].astype('category')
        df[col_name] = df[col_name].cat.codes


# In[ ]:


df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

df


# In[ ]:


sns.swarmplot(x="rating", y="gross", data=df)


# In[ ]:


sns.stripplot(x="rating", y="gross", data=df)


# In[ ]:




