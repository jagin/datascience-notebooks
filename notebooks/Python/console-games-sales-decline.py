
# coding: utf-8

# # Console Games Sales Decline 

# You have been hired to join the in-house Data Science team of a video games design company. The company designs games for computers but is considering getting into the console games business.
# 
# Note: consoles refer to devices that are attached to TV’s. For example, Play Station.
# 
# However, the executives of the company have noticed that other rival console games design companies have been suffering increasing losses in the past couple of years. That’s why they want you to investigate the state of the industry to help them make the decision of whether to get into this business.
# 
# On the Case Study page you will find a dataset with over 16,000 console game titles sold between 1980 and 2015. Sales are broken down into 4 regions and are shown in Millions of Dollars.
# 
# The CSO (Chief Strategy Officer) has posed you the following questions:
# 1.  How have the total sales of console games been declining over the years by different genres?
# 2.  How do different platforms compare side-by-side in terms of aggregate global sales since their inceptions? Who is the leader?
# 3.  How do different publishers compare side-by-side in terms of aggregate global sales since their inceptions? Who is the leader?
# 4.  How do the New Generation (New Gen) consoles compare in terms of total global sales for combined 2014 and 2015? New Gen platforms in this dataset are PS4, XOne and WiiU.
# 5.  What are the top 10 game titles with the highest global sales?

# ## Credits

# This notebook is a part of my learning path based on the workshop [Case Study 001 : [Tableau] Console Games Sales Decline](https://www.superdatascience.com/casestudy001/) presented on [Super Data Science](http://www.superdatascience.com) platform.
# 
# SuperDataScience team publish a lot of courses (available on the [Udemy](https://www.udemy.com) platform) taught by [Kirill Eremenko](https://www.udemy.com/user/kirilleremenko/) and [Hadelin de Ponteves](https://www.udemy.com/user/hadelin-de-ponteves/) in the wide space of Data Science.
# 
# Here are the most valuable one in the field of Data Science, Machine Learning and Deep Learning:
# 1. [Data Science A-Z™: Real-Life Data Science Exercises Included](https://www.udemy.com/datascience/learn/v4/overview)
# 2. [Python A-Z™: Python For Data Science With Real Exercises!](https://www.udemy.com/python-coding/learn/v4/overview)
# 3. [R Programming A-Z™: R For Data Science With Real Exercises!](https://www.udemy.com/r-programming/learn/v4/overview)
# 4. [Machine Learning A-Z™: Hands-On Python & R In Data Science](https://www.udemy.com/machinelearning/learn/v4/overview)
# 5. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/learn/v4/overview)
# 6. [Artificial Intelligence A-Z™: Learn How To Build An AI](https://www.udemy.com/artificial-intelligence-az/learn/v4/overview)

# ## Preparation

# **Import libraries**

# In[1]:


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Use Seaborn styles.
# See 'Controlling figure aesthetics': https://seaborn.pydata.org/tutorial/aesthetics.html
sns.set()
sns.set_style('darkgrid')


# **Load the data**

# In[2]:


df = pd.read_csv("../../data/ConsoleGames.csv")
# See '10 Minutes to pandas': https://pandas.pydata.org/pandas-docs/stable/10min.html


# **Check the structure of the data**
# 
# First, let's get a feel of what our data look like.

# In[3]:


df.head(10)


# Let's see the type of our data, to better understand the structure behind it.

# In[4]:


df.info()


# Let's see the informations on the numerical distribution of our dataset.

# In[5]:


df.describe()


# Let's see the informations on the categorical distribution of our dataset.

# In[6]:


df.describe(include = 'all') # include also non-numerical columns


# In[7]:


def category_describe(data, col) :
    print(data[col].value_counts())
    print(data[col].value_counts().describe())


# In[8]:


category_describe(df, 'Platform')


# In[9]:


category_describe(df, 'Genre')


# In[10]:


category_describe(df, 'Publisher')


# Let's find out which columns has Nan values.

# In[11]:


# checking for missing values
print("Are there missing values? {}".format(df.isnull().any().any()))

df.isnull().sum()


# Let's try to know more about the games with unidentified Publisher

# In[12]:


#Let's affect this specific dataframe, as we will use it several times
publisher_null_df = df[df['Publisher'].isnull()]
publisher_null_df.head(10)


# In[13]:


publisher_null_df.describe()


# Let's drop the null values. As there is not much of them (34 for 15979 in total), it won't make much of a difference.

# In[14]:


df = df.dropna()


# ## 1. How have the total sales of console games been declining over the years by different genres?

# **Create the total sales of console games feature**

# In[15]:


df['Total_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']


# In[16]:


# See 'Explore Happiness Data Using Python Pivot Tables': https://www.dataquest.io/blog/pandas-pivot-table/

df_pivot_table = df.pivot_table(values = 'Total_Sales', index = 'Year', columns = 'Genre', aggfunc = 'sum')
df_pivot_table.plot(kind = 'area', figsize = (12, 6))
plt.title('Total sales over the years')
plt.ylabel('Total sales')
plt.show()


# **Observation**  
# Console games sales have been rapidly declining since 2008. Possibly, this could be attributed to the rising popularity of mobile devices and associated games.

# ## 2. How do different platforms compare side-by-side in terms of aggregate global sales since their inceptions? Who is the leader? 

# In[17]:


df_pivot_table = df.pivot_table(values = 'Total_Sales', index = 'Platform', aggfunc = 'sum').sort_values('Total_Sales')
df_pivot_table = df_pivot_table[df_pivot_table['Total_Sales'] > 5] # filter out platforms with smaller total sales
df_pivot_table.plot(kind = 'barh', legend = None, figsize = (12, 12))
plt.xlabel('Total sales')
plt.show()


# **Observation**  
# The all-time leader in the Global Games Sales among platforms is PS2 making $1.2B+

# ## 3. How do different publishers compare side-by-side in terms of aggregate global sales since their inceptions? Who is the leader? 

# In[18]:


df_pivot_table = df.pivot_table(values = 'Total_Sales', index = 'Publisher', aggfunc = 'sum').sort_values('Total_Sales', ascending = False)
df_pivot_table = df_pivot_table[df_pivot_table['Total_Sales'] > 100] # filter out publishers with smaller total sales
df_pivot_table.plot(kind = 'bar', legend = None, figsize = (12, 6))
plt.ylabel('Total sales')
plt.show()


# In[19]:


# See: https://github.com/laserson/squarify
import squarify

# create a color palette, mapped to Total_Sales
total_sales = df_pivot_table['Total_Sales'].tolist()
cmap = matplotlib.cm.Reds
mini=min(total_sales)
maxi=max(total_sales)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in total_sales]

max_label_size = 15
labels = df_pivot_table.index.tolist()
labels = [(label[:max_label_size] + '..') if len(label) > max_label_size else label for label in labels]
labels = [label + '\n' + '${:,.0f} M'.format(total_sales[idx]) for idx, label in enumerate(labels)]

plt.figure(figsize=(20, 10))
squarify.plot(sizes = total_sales, label = labels, alpha=.8, color=colors)
plt.axis('off')
plt.rc('font', size=16)
plt.show()


# **Observation**  
# In terms of publishers, Nintendo is the leader in Global Games Sales in $1.7B+ terms, however EA, Activision and Sony are leading in quantity of game titles.

# ## 4. How do the New Generation (New Gen) consoles compare in terms of total global sales for combined 2014 and 2015?
# 
# New Gen platforms in this dataset are **PS4**, **XOne** and **WiiU**.

# In[20]:


df_year_filter = df['Year'].between(2014, 2015, inclusive = True)
df_platform_filter = df['Platform'].isin(['PS4', 'XOne', 'WiiU'])
df_pivot_table = df[df_year_filter & df_platform_filter].pivot_table('Total_Sales', index = 'Platform', aggfunc = 'sum')


# In[21]:


df_pivot_table.plot(y = 'Total_Sales', labels = df_pivot_table.index, kind = 'pie',
                    autopct = '%1.1f%%', startangle = 90, legend = False, fontsize = 14, figsize = (12, 12))
plt.axis('off')
plt.show()


# **Observation**  
# The Global Games Sales leader among new Gen consoles is PS4 accounting for more than half of the sales of all three platforms.

# ## 5. What are the top 10 game titles with the highest global sales?

# In[22]:


df_pivot_table = df.pivot_table(values = 'Total_Sales', index = 'Name', aggfunc = 'sum').sort_values('Total_Sales', ascending = False)
df_pivot_table = df_pivot_table.head(10) # filter out 10 top game titles

# create a color palette, mapped to Publisher
#
# unique_games = df.drop_duplicates('Name')[['Name', 'Publisher']].set_index('Name')
# df_pivot_table['Publisher'] = df_pivot_table.index.map(lambda game: unique_games.loc[game]['Publisher'])
# unique_publisher = df_pivot_table.drop_duplicates('Publisher')['Publisher'].tolist()
# cmap = matplotlib.cm.tab20c
# mini=0
# maxi=len(unique_publisher)
# norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
# colors = [cmap(norm(unique_publisher.index(game[1]))) for game in df_pivot_table.values]
#
# df_pivot_table.plot(kind = 'bar', legend = None, color = colors, figsize = (12, 6))

df_pivot_table.plot(kind = 'bar', legend = None, color = 'darkred', figsize = (12, 6))
plt.xlabel('Game title')
plt.ylabel('Total sales')
plt.show()


# **Observation**  
# Top 10 game titles in terms of all-time Global Games Sales.
