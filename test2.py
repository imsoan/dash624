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


# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px


# In[ ]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[ ]:


#url = "https://raw.githubusercontent.com/imsoan/dash624/master/2023-02-08-DATA624-Assignment4-Data.csv"
#df2 = load_data(url)


# In[ ]:


df2 = pd.read_csv("2023-02-08-DATA624-Assignment4-Data.csv")
df2.head()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering


# In[ ]:


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


# In[ ]:


tmp = (pd.concat([
    df,
    pd.DataFrame(
        clusters.labels_,
        columns=['cluster']
    ).astype('category')
], axis=1)
)


# In[ ]:


fig1=px.scatter(tmp, x='5',
     y='9',
     color='cluster',
    symbol='cluster',
 )


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

