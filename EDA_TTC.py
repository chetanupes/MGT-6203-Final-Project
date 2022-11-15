from functools import total_ordering
from os import write
from IPython.core.display import Image
import pandas as pd
import numpy as np
from jinja2 import escape
import joblib

#Streamlit
import streamlit as st
from streamlit.state.session_state import Value
#from streamlit_pandas_profiling import st_profile_report
st.set_page_config(layout="wide")

#Plotting
import plotly.express as px
import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
plt.style.use('seaborn')

#Warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Widgets libraries
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive
from PIL import Image

#Model libraries
#from pandas_profiling import ProfileReport
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xg


#############################################################################################################################################

st.title('MGT-6203 Project (Toronto’s TTC subway delay)')

st.write('The Toronto subway is a rapid transit system serving Toronto and the neighboring city of Vaughan in Ontario, Canada, operated by the Toronto Transit Commission (TTC). It is a multimodal rail network consisting of three heavy-capacity rail lines operating predominantly underground, and one elevated medium-capacity rail line. Two light rail lines, which will operate both at-grade and underground, are under construction.')

#Team
st.title('Team Members')
st.write('1. Tadeus Rossetti Marchesi')
st.write('2. Meaghan Wright')
st.write('3. Chetan Tewari')

st.title('Problem Statement')
st.write('The object of this study is three-fold:')
st.write('1. What can we learn about delays?' )
st.write('2. Can we predict them?')
st.write('3. Can we prevent them?')



# assign dataset names
list_of_names = ['-jan-2014-april-2017','-may-december-2017','-data-2018','-data-2019','-data-2020','-data-2021','-data-2022']
 
# create empty list
df = []

# append datasets into the list
for i in range(len(list_of_names)):
    temp_df = pd.read_excel("ttc-subway-delay"+list_of_names[i]+".xlsx", sheet_name=None)
    temp_df=pd.concat(temp_df.values())
    df.append(temp_df)

merged_df = pd.concat(df)
merged_df.reset_index(inplace=True, drop=True)

# TTC code
code=pd.read_excel('ttc-subway-delay-codes -grouped.xlsx')
code_merged=merged_df.merge(code, left_on='Code', right_on='SUB RMENU CODE')
code_merged.drop(columns='SUB RMENU CODE', inplace=True)

#Data Cleanup
# Remove '0' Min Delay values as they don't provide useful delay info
df_0=code_merged[code_merged['Min Delay']!=0].reset_index(drop=True)

#Creating year and month columns
df_0['Year']=df_0['Date'].dt.strftime('%Y') #Getting year from the data, will use it later for grouping
df_0['Month']=df_0['Date'].dt.strftime('%m')
df_0["Hour"] = df_0['Time'].str.split(':').str[0]

#Creating a season column
col         = 'Month'
conditions  = [df_0[col].str.contains('12|1|2'), df_0[col].str.contains('3|4|5'), 
               df_0[col].str.contains('6|7|8'),df_0[col].str.contains('9|10|11')]
choices     = [ "Winter", 'Spring', 'Summer','Fall' ]
    
df_0["Season"] = np.select(conditions, choices, default=np.nan)

#Creating a new time column
col         = 'Hour'
conditions  = [df_0[col].str.contains('22|23|01|02|03|04|00'), df_0[col].str.contains('05|06|07|08|09|10|11'), 
               df_0[col].str.contains('12|13|14|15'),df_0[col].str.contains('16|17|18|19'),df_0[col].str.contains('20|21')]
choices     = [ "Midnight", 'Morning', 'Afternoon','Evening','Night']
    
df_0["Time_New"] = np.select(conditions, choices, default=np.nan)

#Dropping columns which we don't intent to use in the modeling
df_0=df_0.drop(columns=['Hour','Time','CODE DESCRIPTION','Vehicle'])

#Creating a profile report

#st.title('Dataset Report')
#report=ProfileReport(df_0, title='Profiling Reoprt')

#if st.checkbox('Preview Profile Report'):
    #st_profile_report(report)
#st.write('The profiling function presents the user with a descriptive statistical summary of all the features of the dataset.')

#Vizualization
#Creating some vizulaization
st.title('Creating some visualization')

df_data=copy.deepcopy(df_0)

delays_by_day  = df_data.groupby(['Year','Day']).agg({'Min Delay' : 'sum'}).reset_index().sort_index(ascending=False)
fig = px.bar(delays_by_day, x="Day", y="Min Delay", color='Year')
fig.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Days for each Year")

# Plot!
st.plotly_chart(fig, use_container_width=True)

delays_by_month  = df_data.groupby(['Year','Month']).agg({'Min Delay' : 'sum'}).reset_index().sort_index(ascending=False)
fig1 = px.bar(delays_by_month, x="Month", y="Min Delay", color='Year')
fig1.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Month for each Year")

# Plot!
st.plotly_chart(fig1, use_container_width=True)

df_year=df_data.groupby(['Year']).agg({'Min Delay' : 'sum'}).reset_index().sort_values('Min Delay', ascending=False)
fig2 = px.bar(df_year, x="Year", y="Min Delay")
fig2.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Year")

# Plot!
st.plotly_chart(fig2, use_container_width=True)

delays_by_season  = df_data.groupby(['Year','Season']).agg({'Min Delay' : 'sum'}).reset_index()
fig3 = px.bar(delays_by_season, x="Year", y="Min Delay",color="Season")
fig3.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Season for each year")

# Plot!
st.plotly_chart(fig3, use_container_width=True)

delays_by_code  = df_data.groupby(['Code Group']).agg({'Min Delay' : 'sum'}).reset_index().sort_values('Min Delay',ascending=False)
fig4 = px.bar(delays_by_code, x="Code Group", y="Min Delay")
fig4.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Code Group")

 #Plot!
st.plotly_chart(fig4, use_container_width=True)

delays_by_gro  = df_data.groupby(['Year','Code Group']).agg({'Min Delay' : 'sum'}).reset_index().sort_values('Min Delay',ascending=False)
fig6 = px.bar(delays_by_gro, x="Code Group", y="Min Delay", color='Year')
fig6.update_layout(height=500, width=1300,barmode='group',title_text="Delays by Code Group for each year" )

#Plot!
st.plotly_chart(fig6, use_container_width=True)

delays_by_time  = df_data.groupby(['Time_New']).agg({'Min Delay' : 'sum'}).reset_index().sort_values('Min Delay',ascending=False)
fig7 = px.bar(delays_by_time, x="Time_New", y="Min Delay")
fig7.update_layout(height=500, width=1200,barmode='group',title_text="Delays by Time group" )

#Plot!
st.plotly_chart(fig7, use_container_width=True)

st.title('Few Inference from EDA')

st.write('1. Security, mechanical and human caused delays are the most prevalent causes of delays.')
st.write('2. There does not seem to be a strong visual trend by grouping the Min Delays by Year, although more recent years (2018, 2020-2021) show a slight increasing trend in Min Delays.')
st.write('3. The most delays occurred during the morning, which appears consistent with our hypothesis.')
st.write('4. There is a strong trend of security causing the most delays across all years. There does not seem to be any strong trends grouping the delays by days and month for each year.')
st.write('5. For the present analysis Disorderly Patron was the top reason for delays across most years.')

#######################################################################################################################################################
#Model Building
df_data=copy.deepcopy(df_0)
df_data=df_data.drop(columns=['Date','Code','Bound','Line','Month','Year'])

#Creating dummy variables for Day, station, code group, season and year
station=df_data['Station'].unique()[:-1]
code_group=df_data['Code Group'].unique()[:-1]
season=df_data['Season'].unique()[:-1]
time=df_data['Time_New'].unique()[:-1]
day=df_data['Day'].unique()[:-1]

#Loading the model
xg_model = joblib.load('XG.sav')

# Prediction
# Selecting the method for analysis   
st.sidebar.markdown("# Predictive Model User Input Selection")

# Selecting your options
Select_Station=st.sidebar.selectbox('Select Station', (df_data['Station'].unique()))
Select_Day=st.sidebar.selectbox('Select Day', (df_data['Day'].unique()))
Select_Code=st.sidebar.selectbox('Select Code', (df_data['Code Group'].unique()))
Select_Season=st.sidebar.selectbox('Select Season', (df_data['Season'].unique()))
Select_Time=st.sidebar.selectbox('Select Time', (df_data['Time_New'].unique()))
Select_mingap=st.sidebar.slider('Select MinGap', 5,1000,value=20)

test_dict={'Min Gap':[Select_mingap],'Station':[Select_Station],'Day':[Select_Day],
          'Season':[Select_Season],'Code Group':[Select_Code],'Time_New':[Select_Time]}

test=pd.DataFrame(test_dict)

for i in station:
    test['Station_{}'.format(i)]=np.where(test['Station']==i, 1, 0)
df_1=copy.deepcopy(test)
for i in day:
    df_1['Day_{}'.format(i)]=np.where(df_1['Day']==i, 1, 0)
df_2=copy.deepcopy(df_1)
for i in season:
    df_2['Season_{}'.format(i)]=np.where(df_2['Season']==i, 1, 0)
df_3=copy.deepcopy(df_2)
for i in code_group:
    df_3['Code Group_{}'.format(i)]=np.where(df_3['Code Group']==i, 1, 0)
df_4=copy.deepcopy(df_3)
for i in time:
    df_4['Time_New_{}'.format(i)]=np.where(df_4['Time_New']==i, 1, 0)
pred=copy.deepcopy(df_4)

pred=pred.drop(['Day','Station','Code Group','Season','Time_New'], axis=1)

#Prediction on test
pred_xg1=xg_model.predict(pred)
result=np.round(pred_xg1,2)
###############################################################################################################################################
#Prediction
st.title('Prediction')

st.write('The min delay predicted based on the user selection in minutes is:') 
result

#References
st.title('References')
st.write('[1] Keller, C.; Glück, F.; Gerlach, C.F.; Schlegel, T. Investigating the Potential of Data Science Methods for Sustainable Public Transport. Sustainability 2022, 14, 4211.')
st.write('[2] Burr, T., Merrifield, S., Duffy, D., Griffiths, J., Wright, S., Barker, G., 2008. Reducing Passenger Rail Delays by Better Management of Incidents. Stationery Office, London. ')
st.write('[3] Preston, J., Wall, G., Batley, R., Ibáñez, J.N., Shires, J., 2009. Impact of delays on passenger train services. Transport. Res. Rec.: J. Transport. Res. Board 2117 (1), 14–23.')
st.write('[4] Fredrik Monsuur, Marcus Enoch, Mohammed Quddus, Stuart Meek, Modeling the impact of rail delays on passenger satisfaction, Transportation Research Part A: Policy and Practice, Volume 152, 2021, Pages 19-35, ISSN 0965-8564.')
st.write('[5] Nils O.E. Olsson, Hans Haugland, Influencing factors on train punctuality—results from some Norwegian studies, Transport Policy, Volume 11, Issue 4, 2004, Pages 387-397, ISSN 0967-070X.')
st.write('[6] Wiggenraad, P.B.L., 2001. Alighting and Boarding Times of Passengers at Dutch Railway Stations. Report, TRAIL Research School, Delft.')
st.write('[7] Flier, H., Gelashvili, R., Graffagnino, T., Nunkesser, M. (2009). Mining Railway Delay Dependencies in Large-Scale Real-World Delay Data. In: Ahuja, R.K., Möhring, R.H., Zaroliagis, C.D. (eds) Robust and Online Large-Scale Optimization. Lecture Notes in Computer Science, vol 5868. Springer, Berlin, Heidelberg.')
st.write('[8] Tijs Huisman, Richard J. Boucherie, Nico M. van Dijk, A solvable queueing network model for railway networks and its validation and applications for the Netherlands, European Journal of Operational Research, Volume 142, Issue 1, 2002, Pages 30-51, ISSN 0377-2217.')

