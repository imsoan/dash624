#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[4]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[ ]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[ ]:


url = "https://raw.githubusercontent.com/imsoan/dash624/master/2023-02-08-DATA624-Assignment4-Data.csv"
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


fig2=px.scatter(tmp,
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


fix3=px.line(y = inertia,x = range(1,10), markers=True)


# 4 seemed to be a good number for the number of clusters

# ### K Means

# In[7]:


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


fig4=px.scatter(tmp,
     x='5',
     y='6',
     color='cluster',
     symbol='cluster',
 )


# ##### Silhouette Score

# In[8]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[9]:


get_ipython().run_cell_magic('time', '', 'df = df2\nsscore = []\n\nfor i in range(2,10):\n    kmeans = KMeans(\n        n_clusters = i, # Number of clusters to find\n        init = "k-means++", # How to place the initial cluster centroids,\n        max_iter= 100, # Maximum number of iterations for the algorithm to run\n        tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going\n    ).fit(\n        df\n    )\n    \n    silhouette_avg = silhouette_score(df, kmeans.labels_)\n    sscore.append(silhouette_avg)\n    \n\nfig5= px.line(y = sscore,x = range(2,10), markers=True)\n')


# ##### Silhouette Values

# In[10]:


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
fig6=px.bar(tmp,
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


fig7=px.scatter(tmp,
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
        html.H1("AG"),
        """
       graph

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

