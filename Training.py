
# coding: utf-8

# JIRA AVRO issues workflow (27-July-2018)
#
# # Can we predict when an issue will be resolved?

# We load the various data files and make a preliminary exploratory analysis.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from dateutil.parser import parse
from datetime import timedelta

import random

import pickle

# NOTE: data is in a folder named data and this folder is in the same folder of the script
data_issues=pd.read_json('data/avro-issues.json',lines=True)


data_issues_csv=pd.read_csv('data/avro-issues.csv')


data_issues_transitions=pd.read_csv('data/avro-transitions.csv')

# # Hypothesis: similar histories ~ similar outcomes
#
# Suppose we have in hands a new issue not in the DATA. Call it A. The intuitive idea is to find the issues in the DATA that are most similar to A and then extrapolate (like an average) the future outcome of A. So we need to understand what it means to be similar and which features are determining that. I will not create a model that determines how two issues are similar but I will let the machine learning algorithm find that by itself. For this purpose I will use a RandomForestRegression algorithm and determine the most relevant features.
#
# STEP 1: collect features for the DATA such as:
#     - lifetime (from creation to last status!=Resolved)
#     - Priority: Major/Minor/Trivial...,
#     - issue_type: Bug, Task...,
#     - number_votes,
#     - number_counts,
#     - number of comments
#     - how many different authors,
#     - most frequent author
#     - number_interventions,
#     - *frequency of the interventions (implicitly included= number_interventions/lifetime)*
#     - history of status (not necessarily a change). Ex: 'In progress', 'In progress', 'Patch Available',...
#     - Time evolution. Ex: '2018-01-01:TIME', '2018-02-02:TIME', '2018-03-03:TIME',... ?

# # Lets determine Lifetime of the issues (date_last_status (not Resolved)-date_created)



t_open_resol=[]
l_trans=len(data_issues_transitions)

for i in range(0,l_trans):
    if data_issues_transitions['to_status'][i]=='Resolved':
        j=i
        while data_issues_transitions['to_status'][j]!='Open':
            j-=1
        t_open_resol.append({'index':[j,i],'key': data_issues_transitions['key'][i],
                             'type': data_issues_transitions['issue_type'][i]
                      ,'whenOpen': data_issues_transitions['created'][j],
                             'whenResol':data_issues_transitions['when'][i],# Note that sometimes the issue is Reopened and the Resolution date=NaN
                            'last_stat_notResol':data_issues_transitions['when'][i-1]})



# NOTE: Status :{'Patch Available', 'In Progress', 'Resolved', 'Reopened' and 'Closed'}
#         Examples:
#             - Open->In Progress->...Resolved->Closed (only 1x Resolved)
#             - Open->...->Resolved->Reopened->...->Resolved->Reopened->...->Resolved->Closed (multiple Resolved)
#
#       I will extract multiple histories from the issue with multiple Resolved:
#       - Open->...->Resolved(1)
#       - Open->...->Resolved(1)->...->Resolved(2)
#       - ...etc

# To determine the Lifetime we need to convert the datetime and compute the difference 'whenResol'-'whenOpen'

for t in t_open_resol:
    t1=parse(t['whenOpen'])
    t2=parse(t['last_stat_notResol'])
    delta=t2-t1
    t['lifetime']=delta

# A function that converst datetime format to number of days
def Delta_days(x):
    seconds=x.seconds+x.microseconds/1000
    total=x.days+seconds/(24*3600)
    return total


for t in t_open_resol:
    t['lifetime']=Delta_days(t['lifetime'])


issue_types=[]
for i in t_open_resol:
    j1=i['index'][0]
    j2=i['index'][1]
    a=data_issues_transitions['issue_type'][j1]
    b=data_issues_transitions['issue_type'][j2]
    if a==b:
        issue_types.append(data_issues_transitions['issue_type'][j2])
    else:
        print("something not right!")
issue_types_set=set(issue_types)


# Associate a number: [0,1,..6] to an issue. I will do this randomly to decrease bias.
#



c=[i for i in range(0,7)]
c=random.sample(c,7)
L=['Bug', 'Improvement', 'New Feature', 'Sub-task', 'Task', 'Test', 'Wish']
issue_dic={}
j=0
for i in L:
    issue_dic[i]=c[j]
    j+=1

#need to print this dic to a file because of the random assignment of the integers
np.save('issue_type_dic.npy', issue_dic)

issue_types_num=[]
for i in issue_types:
    issue_types_num.append(issue_dic[i])


priority=[]
for t in t_open_resol:
    i=t['index'][1]
    priority.append(data_issues_transitions['priority'][i])


set(priority)


# In this case I will associate higher values to more difficult issues on the hypothesis that higher difficulty leads to longer time for resolution (according to JIRA documentation website).
#
# Trivial:0
# Minor:1
# Major:2
# Critical:3
# Blocker:4

p_dic={'Trivial':0,'Minor':1,'Major':2,'Critical':3,'Blocker':4}

priority_num=[]
for i in priority:
    priority_num.append(p_dic[i])


#Votes
votes=[]
for t in t_open_resol:
    i=t['index'][1]
    votes.append(data_issues_transitions['vote_count'][i])


#Watch_counts
w_counts=[]
for t in t_open_resol:
    i=t['index'][1]
    w_counts.append(data_issues_transitions['watch_count'][i])


# # Comments_Counts


c_counts=[]
for t in t_open_resol:
    i=t['index'][1]
    c_counts.append(data_issues_transitions['comment_count'][i])



# # Who_counts

who_counts=[]
for t in t_open_resol:
    j=t['index'][0]
    i=t['index'][1]
    aux=[]
    for k in range(j,i): #NOTE: get the data up to the status just before 'Resolved'
        aux.append(data_issues_transitions['who'][k])
    who_counts.append(aux)


who_counts[:10]


# diversity or resilience
#
# - more diversity if the Set={authors} is also bigger
# - more resillience if the average {author1 x n1, author2 x n2,...}-> average(n1,n2,...) is also bigger


diversity=[]
for a in who_counts:
    s=set(a)
    number=len(s)
    diversity.append(number)


resilience=[]
for a in who_counts:
    s=set(a)
    aux=[]
    for b in s:
        c=a.count(b)
        aux.append(c)
    c=np.array(aux)
    m=c.mean()
    resilience.append(m)


# # Number of interventions (not necessarily giving rise to a change of status)
#
# This corresponds roughly to the variable 'total' in DATA['changelog'][i]['total']. However, we need to be careful because there can be issues with 'Reopened' statuses which require a more detailed counting.
# Also we have to count this number only up to/before the status becomes 'Resolved'

# This information is in data_issues. First we create a set {'key'} of the issues that were resolved.

keys=[]
for t in t_open_resol:
    keys.append(t['key'])


interventions=[]
key0='init'
for key in keys:
    if key!=key0:
        i=data_issues.index[data_issues['key']==key]
        i=i[0] #this is the index
        a=data_issues['changelog'][i]['histories'] #it is a list
        l=len(a)
        aux=[]
        for k in range(0,l):
            for s in a[k]['items']: #items field can have many 'fields'
                if s['field']=='status': #corresponds to a change of status
                    aux.append([k,s['toString']])
        for u in aux:
            if u[1]=='Resolved':
                interventions.append(u[0])
    key0=key


# # History of Status Change:
#
# O: Open
# I: In progress
# P: Patch
# R: Resolved
# Re: Reopened
# ...
#
# Ex: I P I P I R Re R Re
#
# How do we encode such a sequence? Because it has variable dimension we can't just add additional columns. Its best to encode the sequence in a number. Make the identification
# O==0
# I==1
# P==2
# R==3
# Re==4
#
# We can use a base-5 numeral system. That is,
#
# X={0,1,2,3,4}5^0+{0,1,2,3,4}5^1+{0,1,2,3,4}5^2+{0,1,2,3,4}5^3+...
# it is equivalent to the sequence
# X=={O,I,P,R,Re},{O,I,P,R,Re},{O,I,P,R,Re},{O,I,P,R,Re},...
#
# Example:
# Sequence: OIPIP -> 0x5^0+1x5^1+2x5^2+1x5^3+2x5^4=5+50+5^3+2x5^4=1430
#
# The map is one-to-one. Since it grows "geometrically", the emphasis is put on the last status since this is the term that contributes the most to the number.

histories=[]
l_trans=len(data_issues_transitions)

for i in range(0,l_trans):
    if data_issues_transitions['to_status'][i]=='Resolved':
        j=i
        aux=[]
        while data_issues_transitions['to_status'][j]!='Open':
            j-=1
            aux.append(data_issues_transitions['to_status'][j])
        histories.append(aux)


# Convert to base-5:

# Status {'Open','Patch Available', 'In Progress', 'Resolved', 'Reopened' and 'Closed'}

base5={'Open':0,'In Progress':1,'Patch Available':2,'Resolved':3,'Reopened':4}



h_base5=[]
for h in histories:
    aux=[]
    for a in h:
        aux.append(base5[a])
    h_base5.append(aux)

#now convert to base-5 numeral system

histories_b5=[]
for h in h_base5:
    c=0
    j=0
    l=len(h)
    for i in range(l-1,-1,-1): #Note that it goes backward according to the assignment in histories[i]
        c+=h[i]*(5**j)
        j+=1
    histories_b5.append(c)



"""
l=len(histories)
c=0
for i in range(0,l):
    c2=len(histories[i])
    if c2>c:
        c=c2

for h in histories:
    if len(h)==c:
        print(h)
"""


# # Target=Resolution Date


# Target=Resolution_Date - Open_Date
#


target=[]

for t in t_open_resol:
    t1=parse(t['whenOpen'])
    t2=parse(t['whenResol'])
    d=t2-t1

    d=Delta_days(d)
    target.append(d)


# # Use RandomForestRegression Algorithm
#
# X=
# - histories_b5
# - interventions
# - resilience
# - diversity
# - c_counts (comments_counts)
# - w_counts (watch_counts)
# - votes (vote_counts)
# - priority_num (priority)
# - issue_types_num
# - lifetime: t_open_resol[i]['lifetime']
#
# Y(target)=
# - target
#
# One has 10 features and 1 target


features=['histories','interventions','resilience','diversity','comments','watch',
          'votes','priority','issue_type','lifetime']


# Now we stack the various features column by column


col=[]


# We need to remove data that has lifetime=0. This data is introducing noise which lowers the performace of the algorithm significantly

l2=len(histories_b5)


seq=[]
for i in range(0,l2):
    if t_open_resol[i]['lifetime']==0:
        seq.append(i)



h_aux=[]
for i in range(0,l2):
    if i not in seq:
        h_aux.append(histories_b5[i])

col.append(np.array(h_aux))


i_aux=[]
for i in range(0,l2):
    if i not in seq:
        i_aux.append(interventions[i])

col.append(np.array(i_aux))



r_aux=[]
for i in range(0,l2):
    if i not in seq:
        r_aux.append(resilience[i])

col.append(np.array(r_aux))



d_aux=[]
for i in range(0,l2):
    if i not in seq:
        d_aux.append(diversity[i])
col.append(np.array(d_aux))



c_aux=[]
for i in range(0,l2):
    if i not in seq:
        c_aux.append(c_counts[i])
col.append(np.array(c_aux))



w_aux=[]
for i in range(0,l2):
    if i not in seq:
        w_aux.append(w_counts[i])
col.append(np.array(w_aux))



v_aux=[]
for i in range(0,l2):
    if i not in seq:
        v_aux.append(votes[i])
col.append(np.array(v_aux))



p_aux=[]
for i in range(0,l2):
    if i not in seq:
        p_aux.append(priority_num[i])
col.append(np.array(p_aux))


is_aux=[]
for i in range(0,l2):
    if i not in seq:
        is_aux.append(issue_types_num[i])
col.append(np.array(is_aux))



lifetime=[]
for t in t_open_resol:
    a=t['lifetime']
    lifetime.append(a)



l_aux=[]
for i in range(0,l2):
    if i not in seq:
        l_aux.append(lifetime[i])
col.append(np.array(l_aux))



X=col[0]



l=len(col)

for i in range(1,l):
    X=np.column_stack((X,col[i]))


#Target

trgt_aux=[]
for i in range(0,l2):
    if i not in seq:
        trgt_aux.append(target[i])
Y=np.array(trgt_aux)


# # RandomForest


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as tts


RF=RandomForestRegressor(max_depth=10,n_estimators=40,random_state=20)


# Split data into train and test sets


X_train, X_test, Y_train, Y_test=tts(X,Y, random_state=0)



RF.fit(X_train,Y_train)

#Now we want to save the model using Pickle

filename='AVRO-RForest-prediction.sav'
pickle.dump(RF, open(filename, 'wb'))

#save some data point to test

x_1=X[30]

np.savetxt('data_point_1.txt', x_1, fmt='%f')

"""
Other Regression Algorithms, analysis and plots

RF.score(X_train,Y_train)


# In[240]:


RF.score(X_test,Y_test)


# Feature_importance

# In[179]:


import matplotlib.pyplot as plt


# In[181]:


def plot_feature_importances(model):
    n_features = 10
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),
    features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


# In[241]:


plot_feature_importances(RF)


# # KNeighborsRegressor

# In[125]:


from sklearn.neighbors import KNeighborsRegressor


# In[288]:


X_train, X_test, Y_train, Y_test=tts(X,Y,random_state=0)


# In[146]:


reg=KNeighborsRegressor(n_neighbors=5)


# In[147]:


reg.fit(X_train,Y_train)


# In[148]:


reg.score(X_train,Y_train)


# In[149]:


reg.score(X_test,Y_test)


# # Linear Regression

# In[242]:


from sklearn.linear_model import LinearRegression


# In[243]:


X_train, X_test, Y_train, Y_test=tts(X,Y,random_state=10)


# In[250]:


lr=LinearRegression()


# In[251]:


X1=np.array(l_aux)


# In[262]:


len(X1)


# In[263]:


len(Y)


# In[252]:


lr.fit(X1,Y)


# In[246]:


lr.score(X_train,Y_train)


# In[247]:


lr.score(X_test,Y_test)


# # Lasso

# In[253]:


from sklearn.linear_model import Lasso


# In[264]:


lasso=Lasso(alpha=0.1)


# In[265]:


lasso.fit(X_train,Y_train)


# In[266]:


lasso.score(X_train,Y_train)


# In[267]:


lasso.score(X_test,Y_test)


# # GradientBoostingRegressor

# In[169]:


from sklearn.ensemble import GradientBoostingRegressor


# In[314]:


GBR=GradientBoostingRegressor(alpha=0.001,n_estimators=200, random_state=34)


# In[315]:


GBR.fit(X_train,Y_train)


# In[316]:


GBR.score(X_train,Y_train)


# In[317]:


GBR.score(X_test,Y_test)

"""
