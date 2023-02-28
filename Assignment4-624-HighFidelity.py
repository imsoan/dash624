#!/usr/bin/env python
# coding: utf-8

# # Assignment 4 High-Fidelity 
# 
# #### - Meghana Kompally

# In[4]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import plotly.express as px


# In[ ]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[ ]:


url = "2023-02-08-DATA624-Assignment4-Data.csv"
df2 = load_data(url)


# In[5]:


df2 = pd.read_csv("2023-02-08-DATA624-Assignment4-Data.csv")
df2.head()


# ### Agglomerative Clustering

# In[33]:


from sklearn.cluster import AgglomerativeClustering


# In[34]:


df = df2
#df = circles
# df = moons

clusters = AgglomerativeClustering(
    n_clusters = 5, # Number of clusters to find
    # linkage="ward", #default
    linkage="single",
).fit(
    df
)
#df.head(5)


# In[38]:


tmp = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1)
)


# Trying compare a couple columns to see if there are any clusters, for the final assignment, I will put in more graphs.

# In[40]:


fig1=px.scatter(tmp, x='5',
     y='9',
     color='cluster',
    symbol='cluster',
 )


# ### DBSCAN

# In[10]:


from sklearn.cluster import DBSCAN


# In[11]:


# df = blobs
df = df2
# df = moons

clusters = DBSCAN(
    # eps= 0.5, # Max distance between two points to assign to same cluster 
    eps= 4, # Max distance between two points to assign to same cluster 
    # eps= 2, # Max distance between two points to assign to same cluster 
).fit(
    df
)


# In[12]:


tmp = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1)
)


# In[13]:


px.scatter(tmp,
     x='1',
     y='5',
     color='cluster',
     symbol='cluster',
 )


# ### Elbow Method

# In[16]:


from sklearn.cluster import KMeans


# In[17]:


get_ipython().run_cell_magic('time', '', 'df = df2\n# df = circles\n# df = moons\ninertia = []\nfor i in range(1,10):\n    kmeans = KMeans(\n        n_clusters = i, # Number of clusters to find\n        init = "k-means++", # How to place the initial cluster centroids,\n        max_iter= 100, # Maximum number of iterations for the algorithm to run\n        tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going\n    ).fit(\n        df\n    )\n    #inertia here is the sum of squared distances of samples to their closest cluster center.\n    inertia.append(kmeans.inertia_)\n')


# In[18]:


px.line(y = inertia,x = range(1,10), markers=True)


# 4 seemed to be a good number for the number of clusters

# ### K Means

# In[19]:


from sklearn.cluster import KMeans


# In[20]:


df = df2
# df = circles
# df = moons

kmeans = KMeans(
    n_clusters = 4, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 75, # Maximum number of iterations for the algorithm to run
    tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df[['6','5',]]
)


# In[21]:


tmp=(pd.concat([
    df,
    pd.DataFrame(
        kmeans.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1))


# In[22]:


px.scatter(tmp,
     x='5',
     y='6',
     color='cluster',
     symbol='cluster',
 )


# ##### Silhouette Score

# In[23]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[24]:


get_ipython().run_cell_magic('time', '', 'df = df2\nsscore = []\n\nfor i in range(2,10):\n    kmeans = KMeans(\n        n_clusters = i, # Number of clusters to find\n        init = "k-means++", # How to place the initial cluster centroids,\n        max_iter= 100, # Maximum number of iterations for the algorithm to run\n        tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going\n    ).fit(\n        df\n    )\n    \n    silhouette_avg = silhouette_score(df, kmeans.labels_)\n    sscore.append(silhouette_avg)\n    \n\npx.line(y = sscore,x = range(2,10), markers=True)\n')


# ##### Silhouette Values

# In[30]:


df = df2

kmeans = KMeans(
    n_clusters = 5, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 100, # Maximum number of iterations for the algorithm to run
    tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df
)

svalues = silhouette_samples(df, kmeans.labels_)

tmp = (
    pd.concat([
        pd.DataFrame(svalues),
        pd.DataFrame(
            kmeans.labels_,
            columns=['cluster']
        ).astype('category')
    ], axis=1)
    .sort_values(['cluster',0])
)
tmp=tmp.reset_index()
px.bar(tmp,
     x=0,
     color='cluster',
     
 )


# ### t-SNE

# In[312]:


from sklearn.manifold import TSNE


# In[324]:


get_ipython().run_cell_magic('time', '', '\n# df = blobs\ndf = df2\n# df = moons\n\nts = TSNE(\n    perplexity=50, # Roughly the "size" of the clusters to look for (original paper\n                   # recommends in the 5-50 range, but in general should be less than\n                   # then number of points in your dataset\n    learning_rate="auto",\n    n_iter=1000,\n    init=\'pca\',\n).fit_transform(df)\n\ndf2.head(5)\n')


# In[325]:


tmp=(pd.concat([
    pd.DataFrame(ts),
    df['6'],
], axis=1))


# In[326]:


px.scatter(tmp,
     x=0,
     y=1,
     color='6',
 )


# In[334]:


get_ipython().run_cell_magic('time', '', '\n# df = blobs\ndf = df2\n# df = moons\n\nts = TSNE(\n    perplexity=50, # Roughly the "size" of the clusters to look for (original paper\n                   # recommends in the 5-50 range, but in general should be less than\n                   # then number of points in your dataset\n    learning_rate="auto",\n    n_iter=1000,\n    init=\'pca\',\n).fit_transform(df)\n')


# Attempting to plot multiple dimensions (comparing more than 2-3 columns) 

# In[332]:


# tmp=(pd.concat([
#     pd.DataFrame(ts),
#     df['6'],
# ], axis=1))


# In[41]:


# px.scatter_3d(tmp,
#      x=0,
#      y=1,
#     z=2,
#      color='6',
#  )


# In[ ]:


app.layout = html.Div(
    [
        html.H1("How do the vital signs and lab results pertaining to high blood pressure (hypertension) vary among various age groups and between the two sexes?"),
        """
        To examine the proportion and prevalence of high blood pressure signs among various age groups and between sexes, we used four vital signs data (Heart Rate (beats per minute) [‘HR’], Systolic BP (mm Hg) [‘SBP’], Diastolic BP (mm Hg) [‘DBP’] and Mean arterial pressure (mm Hg) [‘MAP’]) and four lab results data (Calcium (mg/dL), Glucose (mg/dL), Creatinine (mg/dL) and Oxygen saturation from arterial blood (%)) of 4981 patients from two ICU units. We selected the vital signs that were strongly correlated with high blood pressure. We chose calcium as one of our lab results as it is important for healthy blood pressure, it helps blood vessels tighten and relax when they need to. Low calcium has been known to increase the prevalence of cardiovascular diseases like hypertension [2]. High blood pressure is twice as likely to strike a person with diabetes (high glucose) than a person without diabetes. In fact, a person with diabetes and high blood pressure is four times as likely to develop heart disease than someone who does not have either of the conditions. Furthermore, individuals with high creatinine levels have been known to increase their systolic blood pressure and low levels of saturated oxygen levels have been known to damage arteries by making them less elastic and decreasing blood flow. 
Overall, we would like to identify with help of visualizations like density heat maps, strip plots and violin plots, which category of patients have higher risk of high blood pressure by screening the vital signs and lab test results. We assume that the older population would be at a higher risk compared with the younger groups of patients. We also chose those types of graphs as we found them to best present the data and answer the research question being explored. 

        """,
        dcc.Graph(
            figure=fig1,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="OurFirstFigure",
        ), 
    ]
)
if __name__ == '__main__':
    app.run_server(debug=False)

