#!/usr/bin/env python
# coding: utf-8

# In[234]:


#get_ipython().system('pip install dash')
#get_ipython().system('pip install dash_bootstrap_components')
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[235]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[236]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[237]:


df2 = pd.read_csv("2023-02-08-DATA624-Assignment4-Data.csv")
df2.head()


# In[238]:


labelleddata = pd.read_csv("labelleddata.csv")
labelleddata.head()


# In[239]:


tmp3 = pd.read_csv("SV cluster.csv")
tmp3.head()


# In[240]:


# def load_data(url):
#     """
#     Load data from a shared google drive csv
#     :param url: the shared url string
#     :returns: a pandas dataframe
#     """
#     file_id = url.split("/")[-2]
#     dwn_url = "https://drive.google.com/uc?id=" + file_id
#     df = pd.read_csv(dwn_url)
#     return df
#
# url = ""
# df = load_data(url)
# print(df.shape)
#
#
# # In[ ]:
#
#
# def load_data(url):
#     """
#     Load data from a shared google drive csv
#     :param url: the shared url string
#     :returns: a pandas dataframe
#     """
#     file_id = url.split("/")[-2]
#     dwn_url = "https://drive.google.com/uc?id=" + file_id
#     df = pd.read_csv(dwn_url)
#     return labelleddata
# url = ""
# labelleddata = load_data(url)
# print(labelleddata.shape)
#
#
# # In[ ]:
#
#
# def load_data(url):
#     """
#     Load data from a shared google drive csv
#     :param url: the shared url string
#     :returns: a pandas dataframe
#     """
#     file_id = url.split("/")[-2]
#     dwn_url = "https://drive.google.com/uc?id=" + file_id
#     df = pd.read_csv(dwn_url)
#     return tmp3
# url = ""
# tmp3 = load_data(url)
# print(tmp3.shape)


# ### Agglomerative Clustering

# In[241]:


from sklearn.cluster import AgglomerativeClustering


# In[242]:


df = df2

clusters = AgglomerativeClustering(
    n_clusters = 3, # Number of clusters to find
    # linkage="ward", #default
    linkage="single",
).fit(
    df
)
#df.head(5)


# In[243]:


tmp = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['AG cluster']
    ).astype('category')
], axis=1)
)
#display(tmp)
#tmp.to_csv('AG.csv', index=False)


# In[244]:


fig1=px.scatter(labelleddata, x='5',
     y='2.1',
     color='AG cluster'
    
 )


# In[245]:


fig2=px.scatter(labelleddata, x='5',
     y='3',
     color='AG cluster'
    
 )


# Agglomerative clustering best represented the data in three clusters. The column I noticed that has the most amount of clustering is column 5. It produces clusters with almost all of the other columns. Above are two examples of 5 clustering with another column. 

# ### DBSCAN

# In[246]:


from sklearn.cluster import DBSCAN


# In[247]:


df = df2


clusters = DBSCAN(
    
    eps= 6, # Max distance between two points to assign to same cluster 
    
).fit(
    df
)


# In[248]:


tmp1 = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['DBSCAN cluster']
    ).astype('category')
], axis=1)
)

#tmp.to_csv('DBSCAN.csv', index=False)


# In[249]:


fig3=px.scatter(labelleddata,
     x='1',
     y='5',
     color='DBSCAN cluster'
 )


# For columns 1 and 5, I found that the best max distance to create three distinct cluster

# ### Elbow Method

# In[250]:


from sklearn.cluster import KMeans


# In[251]:


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


# In[252]:


fig4=px.line(y = inertia,x = range(1,10), markers=True)


# 4 seemed to be a good number for the number of clusters based on the elbow method for K means

# ### K Means

# In[253]:


from sklearn.cluster import KMeans


# In[254]:


df = df2

kmeans = KMeans(
    n_clusters = 4, # Number of clusters to find
    init = "k-means++", # How to place the initial cluster centroids,
    max_iter= 500, # Maximum number of iterations for the algorithm to run
    tol=0.0001, # Roughly how much the centroids need to change between iterations to keep going
).fit(
    df
)


# In[255]:


tmp2=(pd.concat([
    df,
    pd.DataFrame(
        kmeans.labels_,
        columns=['K-Means cluster']
    ).astype('category')
], axis=1))
#tmp.to_csv('K-Means.csv', index=False)


# In[256]:


fig5=px.scatter(labelleddata,
     x='0',
     y='5',
     color='K-Means cluster'
     
 )


# ##### Silhouette Score

# In[257]:


from sklearn.metrics import silhouette_samples, silhouette_score


# In[258]:


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

# In[259]:


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

tmp3 = (
    pd.concat([
        pd.DataFrame(svalues),
        pd.DataFrame(
            kmeans1.labels_,
            columns=['SV cluster']
        ).astype('category')
    ], axis=1)
    .sort_values(['SV cluster',0])
)
tmp3=tmp3.reset_index()

#tmp.to_csv('SV.csv', index=False)
fig7=px.bar(tmp3,
     x=0,
     color='SV cluster'
     
 )
#tmp3.to_csv('SV cluster.csv', index=False)


# ### t-SNE

# In[260]:


from sklearn.manifold import TSNE


# In[261]:


df = df2

ts = TSNE(
    perplexity=50, # Roughly the "size" of the clusters to look for (original paper
                   # recommends in the 5-50 range, but in general should be less than
                   # then number of points in your dataset
    learning_rate="auto",
    n_iter=1000,
    init='pca',
).fit_transform(df)


# In[262]:


tmp4=pd.concat([pd.DataFrame(ts), pd.DataFrame(
        kmeans.labels_,
        columns=['tsne cluster']
    ).astype('category')
], axis=1)

#tmp4.to_csv('T-SNE.csv', index=False)
#display(tmp4)


# In[263]:


fig8=px.scatter(tmp4, x=0,y=1, color="tsne cluster")
     


# In[264]:


bigdata=tmp.merge(tmp1,on=(['0','1','2','3','4','5','6','7','8','9','0.1','1.1','2.1','3.1','4.1'])).merge(tmp2,on=(['0','1','2','3','4','5','6','7','8','9','0.1','1.1','2.1','3.1','4.1']))

# frames=[bigdata,tmp4]
# result = pd.concat(frames, keys=["0", "1"])
# #bigdata1=bigdata.merge(tmp4, on=(['1']))
extracted_col1 = tmp3["SV cluster"]
extracted_col = tmp4["tsne cluster"]
bigdata.insert(18, "tsne cluster", extracted_col)
bigdata.insert(19, "SV cluster", extracted_col1)
bigdata1=pd.DataFrame(bigdata)

#display(bigdata1)
#bigdata1.to_csv('labelleddata.csv', index=False)


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



# In[ ]:




