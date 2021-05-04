#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:36:42 2021

@author: aditya
"""



"""

handle
-- change in team name 
-- change in venue name
-- score at 7.1 rather than 5.6

"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'../Data/ipl_csv2/all_matches.csv')
data.columns.tolist()

data = data.fillna(0)
data = data[data['innings'].isin([1,2])]
data['scores'] = data['runs_off_bat']  + data['wides'] + data['noballs'] + data['byes']  + data['legbyes'] + data['penalty'] 

data['venue']  = data['venue'].apply(lambda x : x.split(',')[0])

# 

np.sort(data['venue'].unique())
np.sort(data['batting_team'].unique())
#data['team_score'] = data.groupby(['match_id','innings'])['scores'].transform(lambda x : x.cumsum())

venue_ipl2021 = {'Arun Jaitley Stadium': 'Feroz Shah Kotla',
                 'Narendra Modi Stadium, Motera, Ahmedabad':'Sardar Patel Stadium',
                 'M.Chinnaswamy Stadium' : 'M Chinnaswamy Stadium',
                 'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium'
          } 


team_ipl2021 = {'Deccan Chargers': 'Sunrisers Hyderabad',
                'Delhi Daredevils' :  'Delhi Capitals',
                 'Rising Pune Supergiant' :  'Rising Pune Supergiants'
                 } 

data['venue'] = data['venue'].replace(venue_ipl2021, regex=True)
data['batting_team'] =  data['batting_team'].replace(team_ipl2021, regex=True)
data['bowling_team'] =  data['bowling_team'].replace(team_ipl2021, regex=True)


venue= data[['venue','match_id']].drop_duplicates()['venue'].value_counts().reset_index()
venue_list = venue[venue['venue']>10]['index']



data = data[data['venue'].isin(venue_list)]


tr = int(0.70*data['match_id'].unique().shape[0])
tr_id = data['match_id'].unique()[0:tr]
ts_id = data['match_id'].unique()[tr:819]

train = data[data['match_id'].isin(tr_id)]
test = data[data['match_id'].isin(ts_id)]

train['team_score'] = train.groupby(['match_id','innings'])['scores'].transform(lambda x : x.cumsum())

train = train[(train['ball']==5.6) & (train['innings'].isin(['1','2']))]

venue_score  = train.groupby(['venue','innings','bowling_team'])['team_score'].mean().reset_index().rename(columns = {
        'team_score':'pred_score'})
     

plt.figure(figsize=(10,5))
g = sns.boxplot(x="venue", y="team_score", hue="innings", data=train)
g.set_xticklabels( g.get_xticklabels(), rotation=90 , size = 7)
# Show the plot
plt.show()


test = test.merge(venue_score , on =[ 'venue','innings','bowling_team'] ,how ='left')
test['team_score'] =  test.groupby(['match_id','innings','bowling_team'])['scores'].transform(lambda x : x.cumsum())
pred = test[(test['ball']==5.6) ][['match_id','innings','venue','batting_team','bowling_team','team_score','pred_score']]
((pred['team_score'] - pred['pred_score'])**2/pred['team_score'].sum()).sum()


"""
To do:
    
strike rates -
boundary/innings -
batsman stike rate in first 15 balls - 
bowler economy  -
bowler strike rate in first 2 overs - 
if new give - 60% of best 
    
venues - avg score in first 6 overs


quanitfy attacking batting - batsman sr / match sr ?
quantify wicket taking - bowlers sr / match sr ?
quantify econmoical - bowler eco / match eco ?

prepare test/train data in input format

"""


"""
R squared
-- only venues   3.3228891884282774 
-- only venues and innings  3.329446773970389
-- venues , innings and batting teams  3.199090642640181 -- can only leverage once we predict after toss
-- venues, innings and bowling team 3.4677736945734274
"""

"""

-- {batsman:
    avg score in first 15 balls : s,
    avg match scores : m,
    avg bounday in first 15 balls : b,
    sr in first 15 ballls :  sr,
    boundary scores : bdry_scr
    }
    
--  {bowler :
    
     }
   
"""
data['striker_runs'] = data.groupby(['match_id','innings','batting_team','striker'])['runs_off_bat'].transform(lambda x : x.cumsum())
data['striker_balls_faced'] = data.groupby(['match_id','innings','batting_team','striker']).cumcount() + 1
data['wides_edit'] = data['wides'].apply(lambda x : 1 if x >0 else 0 )
data['wides_cumulative'] = data.groupby(['match_id','innings','batting_team','striker'])['wides_edit'].cumsum()
data['4s_ind'] = data['runs_off_bat'].apply(lambda x : 1 if x==4  else 0)
data['6s_ind'] = data['runs_off_bat'].apply(lambda x : 1 if x ==6 else 0)
data['boundary'] = data['4s_ind'] + data['6s_ind']

# actual balls faced as wides was getting counted in balls faced
data['striker_balls_faced'] = data['striker_balls_faced'] - data['wides_cumulative']

data['striker_boundaries'] = data.groupby(['match_id','innings','batting_team','striker'])['boundary'].cumsum()
data['striker_4s'] = data.groupby(['match_id','innings','batting_team','striker'])['4s_ind'].cumsum()
data['striker_6s'] = data.groupby(['match_id','innings','batting_team','striker'])['6s_ind'].cumsum()


g =  data.groupby(['match_id','innings','batting_team','striker']).cumcount() + 1






















