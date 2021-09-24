#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[128]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
import plotly.express as px

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[129]:


country_df=pd.read_csv("D:/Praachi/Python/Assignments/EDA_Covid/covid_19_india.csv")


# In[130]:


#Let's show the full DataFrame by setting next options prior displaying your data:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# In[131]:


# first 5 rcords
country_df.head()


# In[132]:


# last 5 rcords
country_df.tail()


# In[133]:


# checking rows and columns
country_df.shape


# # Data Exploration and Data Cleaning / Manipulation

# In[134]:


# checking NULL values
country_df.isnull().sum()

# no null values present
# In[135]:


country_df.info()


# In[136]:


# Deleting 3 columns - sno , ConfirmedIndianNational,ConfirmedForeignNational

country_df=country_df.drop(['Sno','ConfirmedIndianNational','ConfirmedForeignNational'],axis=1)


# In[137]:


country_df


# In[138]:


# checking Unique values for column State/UnionTerritory
country_df['State/UnionTerritory'].unique()


# In[139]:


# Replace values with proper naming

country_df=country_df.replace('Bihar****','Bihar')
country_df=country_df.replace('Madhya Pradesh***','Madhya Pradesh')
country_df=country_df.replace('Maharashtra***','Maharashtra')
country_df=country_df.replace('Himanchal Pradesh','Himachal Pradesh')
country_df=country_df.replace('Karanataka', 'karnataka')
country_df=country_df.replace('Telengana', 'Telangana')
#country_df=country_df.replace('Madhya Pradesh', 'MP')
country_df=country_df.replace('karnataka', 'Karnataka')


# In[140]:


# dropping value Cases being reassigned to states for the column State/UnionTerritory

dropn_indexnames =country_df[(country_df['State/UnionTerritory'] == 'Unassigned')].index 
country_df.drop(dropn_indexnames,inplace=True)


# In[141]:


dropn_indexnames =country_df[(country_df['State/UnionTerritory'] == 'Cases being reassigned to states')].index 
country_df.drop(dropn_indexnames,inplace=True)


# In[142]:


# checking Unique values for column State/UnionTerritory
country_df['State/UnionTerritory'].unique()


# In[ ]:





# In[143]:


# Checking Unique values for each column

print("Unique values of each columns: ")
for col in country_df.columns:
  print(f"{col}: \n{country_df[col].unique()}\n")


# In[144]:


country_df.describe()


# In[ ]:





# ## Data Cleaning

# In[216]:


# Create a copy of base data for manupulation & processing

cdf = country_df.copy()


# In[217]:


#lets convert the Date feature to Date&time datatype
cdf['Date']=pd.to_datetime(cdf['Date'],format='%Y-%m-%d')

#Time is not required as it doesnt make much difference
cdf.drop(['Time'],axis=1, inplace=True)

#Renaming State/UnionTerritory to States for easy reference
cdf.rename(columns={'State/UnionTerritory':'States'}, inplace=True)


# In[218]:


cdf.dtypes


# In[219]:


cdf.isnull().sum()


# In[220]:


# Counting active cases

cdf['Active'] = cdf['Confirmed'] - cdf['Cured'] - cdf['Deaths']

cdf["Mortality_Rate"] = np.round(100*cdf["Deaths"]/cdf["Confirmed"],2)
#cdf["Recovery Rate"] = np.round(100*cdf["Cured"]/cdf["Confirmed"],2)
cdf["Recovery_Rate"] = np.round(100*cdf["Cured"]/cdf["Confirmed"],2)


# In[221]:


cdf.dtypes


# In[222]:


#cdf.columns = ['Date', 'States', 'Recovered', 'Deaths', 'Confirmed', 'Active', 'Mortality Rate', 'Recovery Rate', 'Fatality-Ratio']
cdf.rename(columns={'Cured':'Recovered'}, inplace=True)

cdf.head() 


# In[223]:


missing = pd.DataFrame((cdf.isnull().sum())*100/cdf.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot(x='index',y=0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# In[224]:


cdf.isnull().sum()


# In[225]:


#cdf.Mortality_Rate = cdf.Mortality_Rate.fillna(0)
#cdf.Recovery_Rate = cdf.Recovery_Rate.fillna()
cdf.Mortality_Rate.fillna(value=0, inplace = True)
cdf.Recovery_Rate.fillna(value=0, inplace = True)


# In[227]:


cdf.dtypes


# In[228]:


missing = pd.DataFrame((cdf.isnull().sum())*100/cdf.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot(x='index',y=0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# *Here we dont have any missing values now*

# In[ ]:





# In[229]:


cdf_Confirmed=cdf.groupby('States')['Confirmed'].max().sort_values(ascending=False).reset_index()
cdf_Confirmed


# In[ ]:





# In[230]:


statewise = pd.pivot_table(cdf, values=['Confirmed','Recovered', 'Deaths','Recovery_Rate', 'Mortality_Rate'], index='States', aggfunc='max')
#statewise['Recovery Rate'] = statewise['Cured']*100 / statewise['Confirmed']
#statewise['Mortality Rate'] = statewise['Deaths']*100 /statewise['Confirmed']
statewise = statewise.sort_values(by='Confirmed', ascending= False)
statewise.style.background_gradient(cmap='YlOrRd')

# Maharashtra tops this list in all aspects excluding the Mortality rate.

# Considering the number of cases Kerala, Goa, Manipur, Tripura, Arunachal Pradesh, Mizoram, Ladakh and Andaman and Nicobar Islands has done a great job when it comes to recovery rate.

# Similarly in terms of Mortality rate Bihar and Punjab seem to have higher numbers

# Though being second in the country w.r.t. total confirmed cases, Kerala seem to have established the healthcare facility to a higher level which is evident from the lower mortality rate (0.89%).
# In[ ]:





# In[232]:


df_Cured=cdf.groupby('States')['Recovered'].max().sort_values(ascending=False).reset_index()
df_Cured.head(5)


# In[233]:


# Lets Check The Confirmed, Recovered, Death and Active Trolls in India

cdf_confirmed_india=cdf.groupby('Date')['Confirmed'].sum().reset_index()
cdf_cured_india=cdf.groupby('Date')['Recovered'].sum().reset_index()
cdf_death_india=cdf.groupby('Date')['Deaths'].sum().reset_index()
cdf_active_india=cdf.groupby('Date')['Active'].sum().reset_index()


# In[234]:


print("The Confirmed Cases are",cdf_confirmed_india.Confirmed.max())
print("The Recovered Cases are",cdf_cured_india.Recovered.max())
print("The Deaths Cases are",cdf_death_india.Deaths.max())
print("The Active Cases are",cdf_active_india.Active.max())


# In[ ]:





# ## Lets see graphical representation of Confirmed, Recovered, Death and Active Trolls in India - Top5 State-wise

# In[235]:


cdf_Line = cdf.copy()
cdf_Line['Date']=pd.to_datetime(cdf_Line['Date'])
cdf_Line=cdf_Line.groupby(['Date', 'States'], as_index=False).sum({"Recovered": "sum", "Deaths": "sum", "Confirmed": "sum", "Active": "sum"}).reset_index()

cdf_Line['Month_Year'] = cdf_Line['Date'].dt.to_period('M')

cdf_Line


# In[ ]:





# In[236]:


# top Confirmed cases states visualization

fig=plt.figure(figsize=(12,8))
ax=sns.lineplot(data=cdf_Line[cdf_Line['States'].isin(['Kerala','Tamil Nadu','Delhi','Maharashtra','Uttar Pradesh','Karnataka'])],
                x='Date',y='Confirmed',hue='States')
ax.set_title("Confirmed cases by highly affected States \n", size=25)
plt.show()

# Maharashtra, Kerala and Karnataka are among the states which have maximum number of confirmed cases, respectively.
# In[ ]:





# In[237]:


# top Active cases states visualization

fig=plt.figure(figsize=(12,8))
ax=sns.lineplot(data=cdf_Line[cdf_Line['States'].isin(['Kerala','Tamil Nadu','Delhi','Maharashtra','Uttar Pradesh','Karnataka'])],
                x='Date',y='Active',hue='States')
ax.set_title("Active cases by highly affected States \n", size=25)
plt.show()

# Again, Maharashtra, Karnataka and Kerala are among the states which have maximum number of active cases, respectively. 

# But, here we can see the good newst that, after vaccination the graph came down for Active cases, that means vaccination was successful and there was decrease in cases.

# Good news is that, here we can see the there is a drop in actives cases after the vaccination process started from 1st May 21, i.e. from June21.
# In[ ]:





# In[238]:


# top Recovered cases states visualization

fig=plt.figure(figsize=(12,8))
ax=sns.lineplot(data=cdf_Line[cdf_Line['States'].isin(['Kerala','Tamil Nadu','Delhi','Maharashtra','Uttar Pradesh','Karnataka'])],
                x='Date',y='Recovered',hue='States')
ax.set_title("Recovered cases by highly affected States \n", size=25)
plt.show()

# Again, Maharashtra, Kerala and Karnataka are among the states which have maximum number of recovered cases, respectively.
# In[ ]:





# In[239]:


# top Death cases states visualization

fig=plt.figure(figsize=(12,8))
ax=sns.lineplot(data=cdf_Line[cdf_Line['States'].isin(['Kerala','Tamil Nadu','Delhi','Maharashtra','Uttar Pradesh','Karnataka'])],
                x='Date',y='Deaths',hue='States')
ax.set_title("Death cases by highly affected States \n", size=25)
plt.show()

# Among all highly affected states, Maharashtra has maximum number of death cases.
# In[ ]:





# In[240]:


x = np.random.randn(1000, 3)
fig, (ax0, ax1,ax2,ax3) = plt.subplots(1,4,figsize=(15,10))

sns.scatterplot(x=cdf_confirmed_india['Date'], y=cdf_confirmed_india['Confirmed'], data=cdf_confirmed_india,ax=ax0)
ax0.tick_params(axis="x", rotation=50)
ax0.set_title('Total Confirmed Cases\n', fontsize=18)

sns.scatterplot(x=cdf_cured_india['Date'], y=cdf_cured_india['Recovered'], data=cdf_cured_india,ax=ax1)
ax1.tick_params(axis="x", rotation=50)
ax1.set_title('Total Cured Cases\n', fontsize=18)

sns.scatterplot(x=cdf_death_india['Date'], y=cdf_death_india['Deaths'], data=cdf_death_india,ax=ax2)
ax2.tick_params(axis="x", rotation=50)
ax2.set_title('Total Death Cases\n', fontsize=18)

sns.scatterplot(x=cdf_active_india['Date'], y=cdf_active_india['Active'], data=cdf_active_india,ax=ax3)
ax3.tick_params(axis="x", rotation=50)
ax3.set_title('Total Active Cases\n', fontsize=18)


# In[ ]:





# In[241]:


df_Cured=cdf.groupby('States')['Recovered'].max().sort_values(ascending=False).reset_index()
df_Cured.head(5)


# In[242]:


fig = px.choropleth(
    df_Cured,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='States',
    color='Recovered',
    color_continuous_scale='YlGnBu',
    title = 'Cured Ratio(%) per State'
)

fig.update_geos(fitbounds="locations", visible=False)


# # Maharashtra has the highest % of Cured cases..

# In[ ]:





# In[243]:


df_Active=cdf.groupby('States')['Active'].max().sort_values(ascending=False).reset_index()
df_Active.head(5)


# In[244]:


fig = px.choropleth(
    df_Active,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='States',
    color='Active',
    color_continuous_scale='matter',
    title = 'Active Ratio(%) per State'
)

fig.update_geos(fitbounds="locations", visible=False)


# ## From our visualizations we can infer that Maharastra, Kerala and Karnataka are at high risks of having a COVID 3 rd wave. Therefore it is vital to concentrate on such locations in order to remove the risk.

# In[245]:


df_Deaths=cdf.groupby('States')['Deaths'].max().sort_values(ascending=False).reset_index()
df_Deaths.head(5)


# In[246]:


#px.treemap(cdf,path=["States"],values="Deaths",title="Overall States Comparision of deaths")
fig = px.choropleth(
    df_Deaths,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='States',
    color='Deaths',
    color_continuous_scale='BlueRed',
    title = 'Death Ratio(%) per State'
)

fig.update_geos(fitbounds="locations", visible=False)


# In[247]:


cdf.head(5)


# In[248]:


#cdf_confirmed_indiaS = cdf.groupby(['States'],as_index=False).Confirmed.sum().reset_index()

#datasetS = cdf_confirmed_indiaS.sort_values('Confirmed', ascending = False).head(10)

cdf_df=cdf.groupby("States").sum({"Recovered": "sum", "Deaths": "sum", "Confirmed": "sum", "Active": "sum"}).reset_index()
cdf_df.head(5)


# In[352]:


state_top15 = cdf_df.nlargest(15,'Confirmed')
state_top15


# In[357]:


# Position of bars on x-axis
ind = np.arange(15)

# Width of a bar 
width = 0.4

plt.figure(figsize=(15,12))
x = state_top15['States']
y = state_top15['Confirmed']
plt.bar(ind+width/2,y,align='edge',width=width,label="Confirmed", color = 'dodgerblue')
y = state_top15['Recovered']
plt.bar(ind+width,y,align='edge',width=width,label="Recovered")
y = state_top15['Active']
plt.bar(ind+3*width/2,y,align='edge',width=width,label="Active")
y = state_top15['Deaths']
plt.bar(ind+2*width,y,align='edge',width=width,label="Deaths")

plt.xticks(ind + 3*width/2, x)
plt.xticks(rotation=90)
# displaying the title
plt.title("Top 15 affected States-wise Confirmed / Recovered / Active / Death Cases \n",fontsize = 20)
plt.legend();

# We can see from above graphs :-
  # Maharashtra is the state, which faced maximum number of Cases, maximum deaths, maximum number of cured cases, maximum number of Active cases, maximum number of cured cases during this period.
# In[251]:


# Let's Check the Fatality Ratio i.e the severity of the disease

cdf['Fatality-Ratio'] = np.round((cdf['Deaths']/cdf['Confirmed']),2)
cdf['Fatality-Ratio']


# In[252]:


cdf.head(5)


# 

# In[253]:


# Calculating the severity of the disease State-wise
fatality_ratio=cdf.groupby('States')['Fatality-Ratio'].sum().reset_index().sort_values(['Fatality-Ratio'])
fatality_ratio = np.round(fatality_ratio,2)
fig = px.scatter(fatality_ratio.tail(15), x="States", y="Fatality-Ratio",
         size="Fatality-Ratio", color="States", hover_name="States", size_max=60,title='Fatality-Ratio Among States')
fig.show()

# Punjab has the highest Fatality Ratio (severity of disease).
# In[254]:


cdf.head(5)


# In[255]:


cdf.dtypes


# In[274]:


cdf_Ratio = cdf.copy()
cdf_Ratio['Date']=pd.to_datetime(cdf_Ratio['Date'])
cdf_Ratio['Month_Year'] = cdf_Ratio['Date'].dt.to_period('M')
#cdf_Ratio = cdf_Ratio.groupby(['Date'], as_index=False).agg({"Mortality_Rate":"sum", "Recovery_Rate":"sum"})
  #cdf_Ratio = cdf_Ratio.groupby(['Month_Year'], as_index=False).agg({"Mortality_Rate":"sum", "Recovery_Rate":"sum"})


# In[275]:


cdf_Ratio.head(5)


# In[303]:


cdf_Ratio = cdf_Ratio.groupby(['Month_Year'], as_index=False).max({"Mortality_Rate":"sum", "Recovery_Rate":"sum"})


# In[304]:


cdf_Ratio.dtypes


# In[ ]:





# In[308]:


#cdf_S= cdf.groupby(['Date'], as_index=False).sum({"Confirmed":"sum", "Recovered":"sum"})
cdf_S = cdf[['Date','Confirmed', 'Recovered']]
cdf_S['Month'] = cdf_S['Date'].dt.month
#df['StartDate'].dt.to_period('M')
cdf_S['Month_Year'] = cdf_S['Date'].dt.to_period('M')
cdf_S['Year'] = cdf_S['Date'].dt.year
cdf_S


# In[309]:


cdf_S.dtypes


# In[310]:


cdf_S


# In[311]:


cdf_Ss = cdf_S.replace({'Month': {1: 'Winter', 2: 'Spring', 3: 'Spring', 4: 'Summer',5: 'Summer', 6: 'Summer', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}})
cdf_Ss


# In[312]:


cdf_S_grp= cdf_Ss.groupby(['Year', 'Month'], as_index=False).sum({"Confirmed":"sum", "Recovered":"sum"})
cdf_S_grp


# In[313]:


cdf_S_grp.dtypes


# In[314]:


cdf_S_grp_20=cdf_S_grp.loc[cdf_S_grp["Year"]==2020]
cdf_S_grp_21=cdf_S_grp.loc[cdf_S_grp["Year"]==2021]


# In[315]:


cdf_S_grp_20


# In[316]:


cdf_S_grp_21


# In[317]:


plt.figure(figsize=(20,15))
cdf_S_grp_20.plot(x='Month',y=['Confirmed','Recovered'],kind='bar',width=0.7,figsize=(10, 8),color=["red","orange"])
plt.title("Comparing Confirmed v/s Cured cases in 2020 in India Season-wise\n\n Autumn(Oct-Nov)  Monsoon(July-Aug-Sept)  Spring(Feb-Mar)  Summer(Apr-May-June) Winter(Dec-Jan) \n")

# The cases in India were high during October-November (Autumn Season) and the lowest during the April-May-June (Summer season) in year 2020 .
# In[ ]:





# In[318]:


plt.figure(figsize=(20,15))

cdf_S_grp_21.plot(x='Month',y=['Confirmed','Recovered'],kind='bar',width=0.61,figsize=(10, 8),color=["blue","yellow"])
plt.title("Comparing Confirmed v/s Cured cases in 2021 in India Season-wise \n\n Autumn(Oct-Nov)  Monsoon(July-Aug-Sept)  Spring(Feb-Mar)  Summer(Apr-MAy-June) Winter(Dec-Jan) \n")

# The cases in India were high during April-May-June (Summer Season) and no cases October-November (Autumn Season) in year 2021.  We have almost equal ratio of Recovered in comparision with the Confirmed cases.
# In[ ]:





# In[319]:


cdf


# In[320]:


cdf_corrM = cdf.groupby("States").max({"Recovered": "sum", "Deaths": "sum", "Confirmed": "sum"}).reset_index()
cdf_corrM.head(5)


# In[321]:


cdf_corrM['States'].unique()


# In[322]:


dropn_indexnames =cdf_corrM[(cdf_corrM['States'] == 'Dadra and Nagar Haveli') | (cdf_corrM['States'] == 'Daman & Diu')].index 
cdf_corrM.drop(dropn_indexnames,inplace=True)


# In[323]:


#cdf_corrM.reset_index()
cdf_corrM.reset_index(drop=True, inplace=True)


# In[324]:


cdf_corrM.head(5)


# In[325]:


cdf_corrM['States'].unique()


# In[326]:


cdf_corrM.head(5)


# In[327]:


country_df.sort_values(by='State/UnionTerritory')


# In[328]:


country_df_Pop=country_df.groupby("State/UnionTerritory").max({"Cured": "sum", "Deaths": "sum", "Confirmed": "sum"}).reset_index()
country_df_Pop.head(5)


# In[329]:


dropn_indexnames =country_df_Pop[(country_df_Pop['State/UnionTerritory'] == 'Dadra and Nagar Haveli') | (country_df_Pop['State/UnionTerritory'] == 'Daman & Diu')].index 
country_df_Pop.drop(dropn_indexnames,inplace=True)


# In[330]:


country_df_Pop.head(5)


# In[331]:


# Adding a column as Population per State

country_df_Pop['Population'] = [397000,52221000,1504000,34293000,119520000,1179000,28724000,959000,19814000,1540000,67936000,28672000,
                           7300000,13203000,37403000,65798000,35125000,293000,73183,82232000,122153000,3103000,3224000,1192000,
                           2150000,46356334,1504000,29859000,77264000,664000,75695000,37220000,3992000,224979000,11250858,96906000
                           ]
country_df_Pop.head()


# In[332]:


# Calculating Cases/10million
country_df_Pop['Cases/10million'] = (country_df_Pop['Confirmed']/country_df_Pop['Population'])*10000000
country_df_Pop.head(5)


# In[333]:


country_df_Pop.sort_values(by='Cases/10million', ascending=False)


# In[335]:


#Visualization to display the variation in COVID 19 figures in different Indian states

plt.figure(figsize=(14,8), dpi=80)
plt.scatter(country_df_Pop['Confirmed'], country_df_Pop['Cases/10million'], alpha=0.5)

plt.xlabel('Number of confirmed Cases', size=12)
plt.ylabel('Number of cases per 10 million people', size=12)
plt.scatter(country_df_Pop['Confirmed'], country_df_Pop['Cases/10million'], color="red")
for i in range(country_df_Pop.shape[0]):
    plt.annotate(country_df_Pop['State/UnionTerritory'].tolist()[i], xy=(country_df_Pop['Confirmed'].tolist()[i], country_df_Pop['Cases/10million'].tolist()[i]),
                xytext = (country_df_Pop['Confirmed'].tolist()[i]+1.0, country_df_Pop['Cases/10million'].tolist()[i]+12.0), size=11)
plt.tight_layout()    
plt.title('Visualization to display the variation in COVID 19 figures in different Indian states \n', size=16)
plt.show()


# So from the visualization, we realise that even if we take the state population in consideration, Maharashtra, Kerala, Karnataka and Tamil Nadu are badly hit indeed. In addition to these states come up other states like Andhra Pradesh, Uttar Pradesh, West Bengal, Delhi and Chandigarh. However, things do look good for Lakshadweep, Chandigarh, Ladakh, Skkkim, Mizoram where many cases have recovered.
# 
# ## OBSERVATION:-  One observation here is, Maharashtra, Kerala, Karnataka & Tamil Nadu are near Coastal  States.

# In[ ]:





# In[336]:


# we can have a look at how these features are co-related to each other 

plt.figure(figsize = (12,8))
sns.heatmap(country_df_Pop.corr(), annot=True)

# We notice that some measures like Confirmed, Cured, Deaths and Cases/10 million are very much co-related.
# Population & Cases/10 million are -vely correlated.
# In[374]:


sns.set(style="ticks")
sns.pairplot(country_df_Pop[["Confirmed","Cured","Deaths","Cases/10million", "Population"]])


# In[337]:



plt.figure(figsize = (12,8))
sns.heatmap(cdf_corrM.corr(), annot=True,  cmap = 'PiYG')

1. Recovered & confirmed rate highly correlated, also Active & Recovered are higly correlated. Fatality Ratio & Mortality Rate are highly correlated.  
2. Recovery Rate & Deaths are -vely correlated, that means when recovery rate increases the mortality rate will be decreased.  
# In[377]:


sns.set(style="ticks")
sns.pairplot(cdf_corrM[["Active","Mortality_Rate","Recovery_Rate","Fatality-Ratio"]])


# ## CONCLUSIONS :- 
RISK ANALYSIS STATES-WISE : 
# Maharashtra is the state, which faced maximum number of Cases, maximum deaths, maximum number of cured cases, maximum number of Active cases, maximum number of cured cases during this period. That means, Maharshtra tops this list in all aspects excluding the Mortality rate. Karnataka, Kerala are also badly hit by the wave.

# In terms of Mortality rate Bihar and Punjab seem to have higher numbers

# Punjab has the highest Fatality Ratio (severity of disease).

# From our visualizations we can infer that Maharastra, Kerala and Karnataka are at high risks of having a COVID 3 rd wave. Therefore it is vital to concentrate on such locations in order to remove the risk.


SOME GOOD INSIGHTS :- 
# At the same time, we can observe that Maharashtra has the highest % of Cured cases.

# Considering the number of cases Kerala, Goa, Manipur, Tripura, Arunachal Pradesh, Mizoram, Ladakh and Andaman and Nicobar Islands has done a great job when it comes to recovery rate.

# Though being second in the country w.r.t. total confirmed cases, Kerala seem to have established the healthcare facility to a higher level which is evident from the lower mortality rate (0.89%).

SEASON-WISE ANALYSIS : 
# 2020: The cases in India were high during October-November (Autumn Season) and the lowest during the April-May-June (Summer season) in year 2020 .
# 2021: The cases in India were high during April-May-June (Summer Season) and no cases October-November (Autumn Season) in year 2021.

# # NOTES :-
#     
# So from the visualization, we realise that even if we take the state population in consideration, Maharashtra, Kerala, Karnataka and Tamil Nadu are badly hit indeed.
# In addition to these states come up other states like Andhra Pradesh, Uttar Pradesh, West Bengal, Delhi and Chandigarh.
# However, things do look good for Lakshadweep, Chandigarh, Ladakh, Skkkim, Mizoram where many cases have recovered.
# 
# We notice that some measures like Confirmed, Cured, Deaths and Cases/10 million are very much co-related.
# Recovered & confirmed rate highly correlated.  Recovery Rate & Deaths are -vely correlated, that means when recovery rate increases the moratlity rate will be decreased. 

# In[ ]:




