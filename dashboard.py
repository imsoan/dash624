#!/usr/bin/env python
# coding: utf-8

# In[5]:


from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
# import matplotlib.pyplot as plt
# import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


# In[6]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[7]:


def load_data(url):
    """
    Load data from a shared google drive csv
    :param url: the shared url string
    :returns: a pandas dataframe
    """
    file_id = url.split("/")[-2]
    dwn_url = "https://drive.google.com/uc?id=" + file_id
    df = pd.read_csv(dwn_url)
    return df


# In[8]:


url = ""
df = load_data(url)
print(df.shape)


# In[9]:


df_updated = df[["patient",'Age', 'Gender', "SBP", "DBP","HR", "MAP","Resp", "SaO2","Calcium","Glucose","Creatinine"]] 


# In[10]:


df1=df_updated.groupby("patient", as_index=False).mean()
df1=pd.DataFrame(df1)
df1.head(10)
df1['patient'].nunique()


# In[11]:


df1.dropna(inplace=True)


# In[12]:


df1 = df1.replace({'Gender' : {0 : 'Male', 1 : 'Female'}})


# In[13]:


# make age groups  
df1.loc[(df1['Age'] >= 18) & (df1['Age'] <= 39),'Age_group'] = '18-39'
df1.loc[(df1['Age'] > 40) & (df1['Age'] <= 60),'Age_group'] = '40-60'  
df1.loc[(df1['Age'] >= 61),'Age_group'] = '61 or older'


# In[14]:


#Figure 1
# Ploting Graphs for Vital signs and comparing the values for both the genders
fig1 = make_subplots(rows=2, cols=2, subplot_titles=("SBP(mm Hg)", "DBP(mm Hg)", "MAP(mm Hg)", "HR(mm Hg)"))
fig1.add_trace(go.Histogram2d(x=df1['Gender'],y=df1['SBP'],colorscale='pubu',texttemplate= "%{z}"),1,1)
fig1.add_trace(go.Histogram2d(x=df1['Gender'],y=df1['DBP'],colorscale='pubu',texttemplate= "%{z}"),1,2)
fig1.add_trace(go.Histogram2d(x=df1['Gender'],y=df1['MAP'],colorscale='pubu',texttemplate= "%{z}"),2,1)
fig1.add_trace(go.Histogram2d(x=df1['Gender'],y=df1['HR'],colorscale='pubu',texttemplate= "%{z}"),2,2)
fig1.update_layout(title="Figure 1. Vital signs related to hypertension between the sexes")
fig1.update_layout(width=800,height=1000)


# In[11]:


#Figure 2
# fig, axes = plt.subplots(2,2,figsize=(13, 9.5))
# sns.set_style("darkgrid")
# plt.subplots_adjust(wspace=0.5, hspace=0.1)
# ax=sns.stripplot(data=df1, x='Gender', y="Calcium", hue='Gender',order=[ "Male", "Female"],ax = axes[0,0])
# ay=sns.stripplot(data=df1, x='Gender', y="Glucose", hue='Gender',order=[ "Male", "Female"],ax = axes[0,1])
# ar=sns.stripplot(data=df1, x='Gender', y="Creatinine", hue='Gender',order=["Male", "Female"],ax = axes[1,0])
# ak=sns.stripplot(data=df1, x='Gender', y="SaO2", hue='Gender',order=[ "Male", "Female"],ax = axes[1,1])
# #title
# #axes[0,0].set_title ("Figure 2. Lab test results related to hypertension between the sexes")

# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Gender")
# sns.move_legend(ay, "upper left", bbox_to_anchor=(1, 1), title="Gender")
# sns.move_legend(ar, "upper left", bbox_to_anchor=(1, 1), title="Gender")
# sns.move_legend(ak, "center left", bbox_to_anchor=(1, 0.5), title="Gender")

# #axis
# ax.set(xlabel = None, ylabel = 'Calcium (mg/dL)')
# ay.set(xlabel = None, ylabel = 'Serum glucose (mg/dL)')
# ar.set(xlabel = None, ylabel = 'Creatinine (mg/dL)')
# ak.set(xlabel = None, ylabel = 'Oxygen saturation from arterial blood (%)')

# ax.axhline(y=8.5,linewidth=0.4, color='#0c2c84',linestyle='--')
# ax.axhline(y=10.2,linewidth=0.4, color='#0c2c84',linestyle='--')
# ax.text(1.5, 9.5,'normal level range', fontsize=8) #add text
# ax.text(0.2,7.5,'63.4%', fontsize=8) #add text
# ax.text(0.15,5.3,'below the normal level', fontsize=8) #add text
# ax.text(1.2,7.5,'65.7%', fontsize=8) #add text


# ay.axhline(y=140,linewidth=0.5, color='#0c2c84',linestyle='--')
# ay.text(1.5, 100,'normal level range 140↓', fontsize=8) #add text
# ay.text(0.2,200,'33.6%', fontsize=8) #add text
# ay.text(1.2,200,'31.0%', fontsize=8) #add text

# ar.axhline(y=0.74,linewidth=0.5, color='#0c2c84',linestyle='--')
# ar.axhline(y=1.35,linewidth=0.5, color='#0c2c84',linestyle='--')
# ar.axhline(y=0.59,linewidth=0.5, color='#FFA500',linestyle='--')
# ar.axhline(y=1.04,linewidth=0.5, color='#FFA500',linestyle='--')
# ar.text(1.5, 1.3,'normal level range (M)', fontsize=8) #add text
# ar.text(1.5, 0.7,'normal level range (F)', fontsize=8) #add text
# ar.text(0.2,3.0,'20.3%', fontsize=8) #add text
# ar.text(1.2,3.0,'46.5%', fontsize=8) #add text

# ak.axhline(y=94,linewidth=0.5, color='#0c2c84',linestyle='--')
# ak.text(1.5, 94,'normal level range 94% ↑', fontsize=8) #add text
# ak.text(0.2,85,'38.7%', fontsize=8) #add text
# ak.text(1.2,85,'37.7%', fontsize=8) #add text

# plt.show()


# In[12]:


#Figure 3
# fig, axes = plt.subplots(4,1,figsize=(4, 18))
# sns.set_style("darkgrid")
# ax=sns.violinplot(data=df1, x='Age_group', y="Calcium", hue='Age_group',order=[ "18-39", "40-60", "61 or older"],width=0.9, ax = axes[0])
# ay=sns.violinplot(data=df1, x='Age_group', y="Glucose", hue='Age_group',order=[ "18-39", "40-60", "61 or older"], width=0.9,ax = axes[1])
# ar=sns.violinplot(data=df1, x='Age_group', y="Creatinine", hue='Age_group',order=[ "18-39", "40-60", "61 or older"],width=0.9, ax = axes[2])
# ak=sns.violinplot(data=df1, x='Age_group', y="SaO2", hue='Age_group',order=[ "18-39", "40-60", "61 or older"], width=0.9, ax = axes[3])
# #title
# axes[0].set_title ("Figure 3. Lab test results related to hypertension among various age groups")
# #legend
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Age group")
# sns.move_legend(ay, "upper left", bbox_to_anchor=(1, 1), title="Age group")
# sns.move_legend(ar, "upper left", bbox_to_anchor=(1, 1), title="Age group")
# sns.move_legend(ak, "upper left", bbox_to_anchor=(1, 1), title="Age group")

# #axis
# ax.set(xlabel = None, ylabel = 'Calcium (mg/dL)')
# ay.set(xlabel = None, ylabel = 'Serum glucose (mg/dL)')
# ar.set(xlabel = None, ylabel = 'Creatinine (mg/dL)')
# ak.set(xlabel = None, ylabel = 'Oxygen saturation from arterial blood (%)')

# #lines indication normal level for each lab test
# ax.axhline(y=8.5,linewidth=0.4, color='#0c2c84',linestyle='--')
# ax.axhline(y=10.2,linewidth=0.4, color='#0c2c84',linestyle='--')
# ax.text(1.9, 9.5,'normal level range', fontsize=8) #add text
# ax.text(0.2,7.5,'69.4% patients', fontsize=8) #add text
# ax.text(0.1,7,'below the normal level', fontsize=8) #add text
# ax.text(1.35,6,'65.2%', fontsize=8) #add text
# ax.text(1.8,6,'64.4%', fontsize=8) #add text

# ay.axhline(y=140,linewidth=0.5, color='#0c2c84',linestyle='--')
# ay.text(1.9, 100,'normal level range 140↓', fontsize=8) #add text
# ay.text(0.1,200,'26.7%', fontsize=8) #add text
# ay.text(1.35,200,'35.3%', fontsize=8) #add text
# ay.text(1.8,200,'31.3%', fontsize=8) #add text

# ar.axhline(y=0.84,linewidth=0.5, color='#0c2c84',linestyle='--')
# ar.axhline(y=1.21,linewidth=0.5, color='#0c2c84',linestyle='--')
# ar.text(1.9, 1.5,'normal level range', fontsize=8) #add text
# ar.text(0.1,5,'19.4%', fontsize=8) #add text
# ar.text(1.35,5,'18.7%', fontsize=8) #add text
# ar.text(1.8,5,'13%', fontsize=8) #add text

# ak.axhline(y=94,linewidth=0.5, color='#0c2c84',linestyle='--')
# ak.text(1.2, 105,'normal level range 94% ↑', fontsize=8) #add text
# ak.text(0.1,80,'20.7%', fontsize=8) #add text
# ak.text(1.35,80,'18%', fontsize=8) #add text
# ak.text(1.8,80,'21%', fontsize=8) #add text

# plt.show()


# In[16]:


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


# In[ ]:




