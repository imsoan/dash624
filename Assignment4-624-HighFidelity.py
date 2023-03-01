#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install dash')
# get_ipython().system('pip install dash_bootstrap_components')
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[3]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[4]:


df2 = pd.read_csv("2023-02-08-DATA624-Assignment4-Data.csv")
df2.head()


# ### Agglomerative Clustering

# In[5]:


from sklearn.cluster import AgglomerativeClustering


# In[6]:


df = df2
#df = circles
# df = moons

clusters = AgglomerativeClustering(
    n_clusters = 3, # Number of clusters to find
    # linkage="ward", #default
    linkage="single",
).fit(
    df
)
#df.head(5)


# In[7]:


tmp = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1)
)
#display(tmp)
#tmp['cluster'].to_csv('Clusters.csv', index=False)


# In[8]:


fig1=px.scatter(tmp, x='5',
     y='2.1',
     color='cluster',
    symbol='cluster',
 )


# In[9]:


fig2=px.scatter(tmp, x='5',
     y='3',
     color='cluster',
    symbol='cluster',
 )


# Agglomerative clustering best represented the data in three clusters. The column I noticed that has the most amount of clustering is column 5. It produces clusters with almost all of the other columns. Above are two examples of 5 clustering with another column. 

# ### DBSCAN

# In[10]:


from sklearn.cluster import DBSCAN


# In[11]:


df = df2


clusters = DBSCAN(
    
    eps= 6, # Max distance between two points to assign to same cluster 
    
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


fig3=px.scatter(tmp,
     x='1',
     y='5',
     color='cluster',
     symbol='cluster',
 )


# For columns 1 and 5, I found that the best max distance to create three distinct cluster

# ### Elbow Method

# In[14]:


from sklearn.cluster import KMeans


# In[15]:


df = df2

inertia = []
for i in range(1,10):
    kmeans = KMeans(
        n_clusters = i, 
        init = "k-means++", 
        max_iter= 100, 
        tol=0.0001, 
    ).fit(
        df
    )
    
    inertia.append(kmeans.inertia_)


# In[16]:


fig4=px.line(y = inertia,x = range(1,10), markers=True)


# 4 seemed to be a good number for the number of clusters based on the elbow method for K means

# ### K Means

# In[17]:


from sklearn.cluster import KMeans


# In[18]:


df = df2

kmeans = KMeans(
    n_clusters = 4, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 500, # Maximum number of iterations for the algorithm to run
    tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df
)


# In[19]:


tmp=(pd.concat([
    df,
    pd.DataFrame(
        kmeans.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1))


# In[20]:


fig5=px.scatter(tmp,
     x='0',
     y='5',
     color='cluster',
     symbol='cluster',
 )


# ##### Silhouette Score

# In[21]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[22]:


df = df2
sscore = []

for i in range(2,10):
    kmeans1 = KMeans(
        n_clusters = i, # Number of clusters to find
        init = "k-means++", # How to place the initial cluster centroids,
        max_iter= 100, # Maximum number of iterations for the algorithm to run
        tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
    ).fit(
        df
    )
    
    silhouette_avg = silhouette_score(df, kmeans1.labels_)
    sscore.append(silhouette_avg)
    

fig6=px.line(y = sscore,x = range(2,10), markers=True)


# ##### Silhouette Values

# In[23]:


df = df2

kmeans1 = KMeans(
    n_clusters = 7, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 100, # Maximum number of iterations for the algorithm to run
    tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df
)

svalues = silhouette_samples(df, kmeans1.labels_)

tmp = (
    pd.concat([
        pd.DataFrame(svalues),
        pd.DataFrame(
            kmeans1.labels_,
            columns=['cluster']
        ).astype('category')
    ], axis=1)
    .sort_values(['cluster',0])
)
tmp=tmp.reset_index()
fig7=px.bar(tmp,
     x=0,
     color='cluster'
     
 )


# ### t-SNE

# In[24]:


from sklearn.manifold import TSNE


# In[25]:


df = df2

ts = TSNE(
    perplexity=50, # Roughly the "size" of the clusters to look for (original paper
                   # recommends in the 5-50 range, but in general should be less than
                   # then number of points in your dataset
    learning_rate="auto",
    n_iter=1000,
    init='pca',
).fit_transform(df)


# In[26]:

tmp4=pd.concat([pd.DataFrame(ts), pd.DataFrame(
        kmeans.labels_,
        columns=['tsne cluster']
    ).astype('category')
], axis=1)



# In[27]:


fig8=px.scatter(tmp4, x=0,y=1, color="tsne cluster")



# In[ ]:


app.layout = html.Div(
    [
        html.H1("Assignment 4"),
       html.H3("Agglomerative Clustering")
        ,
        dcc.Graph(
            figure=fig1,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="FirstFigure",

        ),


        dcc.Graph(
            figure=fig2,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="SecondFigure",
        ),
        """
              Agglomerative clustering best represented the data in three clusters. The column I noticed that has the most amount of clustering is column 5. It produces clusters with almost all of the other columns. Above are two examples of 5 clustering with another column. 
        """,
        html.H3("DBSCAN")
        ,
        dcc.Graph(
            figure=fig3,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="ThirdFigure",

        ),
        """
            Best max distance to create three distinct clusters
        """,
        html.H3("K-Means (Elbow Method")
        ,
        dcc.Graph(
            figure=fig4,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="FourthFigure",

        ),
        """
            4 seemed to be a good number for the number of clusters based on the elbow method for K means
        """,
        html.H3("K-Means")
        ,

        dcc.Graph(
            figure=fig5,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="FifthFigure",

        ), 

       html.H3("Silhouette Score")
        ,
        dcc.Graph(
            figure=fig6,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="SixthFigure",
       
        ),
        """
            6 seemed to be a good number for the number of clusters based on the silhouette score
        """,
        html.H3("Silhouette Values")
        ,
        dcc.Graph(
            figure=fig7,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="SeventhFigure",
    
        ), 
        html.H3("T-SNE")
        ,
        dcc.Graph(
            figure=fig8,
            style={
                "width": "80%",
                "height": "70vh",
            },
            id="EighthFigure",
        )
    ]
)
if __name__ == '__main__':
    app.run_server(debug=False)

#Dash Web (local):  http://127.0.0.1:8050/



# In[ ]:




