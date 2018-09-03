#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3rd Exercise

@author: JoaoGomes
"""

from flask import Flask, jsonify
from flask import abort
from flask import make_response

import pickle
import numpy as np
import pandas as pd

from dateutil.parser import parse
from datetime import timedelta

import json

app = Flask(__name__)


filename='AVRO-RForest-prediction.sav' #where the model was saved

loaded_model = pickle.load(open(filename, 'rb')) #loads the model

print("Loading data...")
# NOTE: data is in a folder named data and this folder is in the same folder of the script

data_issues_transitions=pd.read_csv('data/avro-transitions.csv')

data_issues_csv=pd.read_csv('data/avro-issues.csv')

data_issues=pd.read_json('data/avro-issues.json',lines=True)

print("Data is loaded.")


def Delta_days(x):
    seconds=x.seconds+x.microseconds/1000
    total=x.days+seconds/(24*3600)
    return total

def prepare_data(i): #i: index of row in data_issues_csv
    #it puts out an array which we can use to predict a target
  
    
    key=data_issues_csv['key'][i]
    j=data_issues_transitions.index[data_issues_transitions['key']==key]
    j=list(j)
    
    l=len(j)
    k1=j[0]
    k2=j[l-1]
    t_data={'key': key,
        'type': data_issues_transitions['issue_type'][k1],
        'whenOpen': data_issues_transitions['created'][k1],
        'current':data_issues_transitions['when'][k2]}
    t1=parse(t_data['whenOpen'])
    t2=parse(t_data['current'])
    delta=t2-t1
    t_data['lifetime']=Delta_days(delta)
    
    issue_type=data_issues_transitions['issue_type'][k1]
    issue_dic=np.load('issue_type_dic.npy').item() #loads the dictionary
    issue_num=issue_dic[issue_type]
    
    priority=data_issues_transitions['priority'][k1]
    p_dic={'Trivial':0,'Minor':1,'Major':2,'Critical':3,'Blocker':4}
    priority_num=p_dic[priority]
    #######
    votes=data_issues_transitions['vote_count'][k1]
    #####
    w_counts=data_issues_transitions['watch_count'][k1]
    #####
    c_count=data_issues_transitions['comment_count'][k1]
    ####
    who=[]
    for u in range(k1,k2+1):
        who.append(data_issues_transitions['who'][u])
    a=set(who)
    diversity=len(a)
    
    aux=[]
    for b in a:
        c=who.count(b)
        aux.append(c)
    c=np.array(aux)
    m=c.mean()
    resilience=m
    
    #######
    
    interventions=[]
    u=data_issues.index[data_issues['key']==key] 
    u=u[0] #this is the index
    status=data_issues_csv['status'][i]
    a=data_issues['changelog'][u]['histories'] #it is a list
    l=len(a)
        
    for k in range(0,l):
        for s in a[k]['items']: #items field can have many 'fields'
            if s['field']=='status': #corresponds to a change of status
                if s['toString']==status:
                    k2=k #counts the number of interventions until the current status is reached
    interventions=k2
    
    ########
    
    histories=[]
    for u in range(k1,k2+1):
        histories.append(data_issues_transitions['to_status'][u])
    base5={'Open':0,'In Progress':1,'Patch Available':2,'Resolved':3,'Reopened':4}
    
    aux=[]
    for a in histories:
        aux.append(base5[a])
    c=0
    j3=0
    l=len(histories)
    for n in range(l-1,-1,-1): #Note that it goes backward according to the assignment in histories[i]
        c+=aux[n]*(5**j3)
        j3+=1
    
    histories_b5=c
    
    #####################
    col=[]
    
    col.append(histories_b5)
    col.append(interventions)
    col.append(resilience)
    col.append(diversity)
    col.append(c_count)
    col.append(w_counts)
    col.append(votes)
    col.append(priority_num)
    col.append(issue_num)
    col.append(t_data['lifetime'])
    
    col=np.array(col)
    col=col.reshape(1,-1)
    
    return col


def days_to_datetime(d):
    #convert days to int days, int seconds, int microseconds
    days=int(d)
    seconds=(d-days)*24*3600
    sec=int(seconds)
    micro_seconds=(seconds-sec)*1000
    micro_sec=int(micro_seconds)
    
    delta=timedelta(days,sec,micro_sec)
    
    return delta


@app.route('/api/release/<date>/resolved-since-now', methods=['GET'])

#name is a date not necessarily in ISOformat

def get_task(date):
    if len(date) == 0:
        abort(404)
    #define the data of json extraction. I will take the date of two weeks ago.
    
    date_extract= '2018-07-16T15:00:00.000+0000'
    
    lst=[]
    s=str()
    l=len(date)
    
    for u in range(0,l):     
        if date[u]!='-':
            s+=date[u]
            if u==l-1:
                lst.append(s)
        else:
            lst.append(s)
            s=str()
    
    #lst=[year,month,day,...]
    date_list=[]
    for a in lst:
        date_list.append(int(a)) #transform the date string into an integer
        
    #get issues that are not resolved
    
    l=len(data_issues_csv)
    lst_non_resol=[]
    
    for a in range(0,l):
        if data_issues_csv['status'][a]!='Resolved':
            lst_non_resol.append(a)
    
    resolved=[]
    for i in lst_non_resol:
        x=prepare_data(i) #the input
        y_predict=loaded_model.predict(x)
        days_predict=y_predict[0] #just one point
        date_current_status=data_issues_csv['updated'][i]
        date_cur=parse(date_current_status)
        date_resol=date_cur+days_to_datetime(days_predict)
        
        if len(date_list)==3: #there should be a match between years, month and days
            if date_resol.year==date_list[0] and date_resol.month==date_list[1] and date_resol.day==date_list[2]:
                key=data_issues_csv['key'][i]
                date_string=date_resol.isoformat()
                resolved.append({'key':key,'predicted_resolution_date':date_string})
        if len(date_list)==2: #only month and year are required to match
            if date_resol.year==date_list[0] and date_resol.month==date_list[1]:
                key=data_issues_csv['key'][i]
                date_string=date_resol.isoformat()
                resolved.append({'issue':key,'predicted_resolution_date':date_string})
    if len(resolved)==0:
        abort(404)
    else:
        task={'now':date_extract, 'issues': resolved}
        
        
    return jsonify(task)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Error :('}), 404)

if __name__ == '__main__':
    app.run(debug=True)
