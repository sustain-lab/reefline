import string
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pickle
import os
from ipywidgets import interact, fixed  #SliderStyle,
from prettytable import PrettyTable

water_depth = 0.43 #this is hard coded right now 
# all 30-second runs
runs_start = [

    datetime(2022, 3, 4, 19, 5,  0), # a = 0.03, f = 0.64, orthogonal
    datetime(2022, 3, 4, 19, 9,  0), # a = 0.06, f = 0.45, orthogonal
    datetime(2022, 3, 4, 19, 13,  0), # a = 0.06, f = 0.32, 10 deg. oblique
    datetime(2022, 3, 4, 19, 17,  0), # a = 0.06, f = 0.32, 10 deg. oblique
    datetime(2022, 3, 4, 19, 21,  0), # a = 0.03, f = 0.64, 30 deg. oblique
    datetime(2022, 3, 4, 19, 25,  0), # a = 0.06, f = 0.45, 30 deg. oblique
    datetime(2022, 3, 4, 19, 29,  0), # a = 0.06, f = 0.45, 30 deg. oblique
    datetime(2022, 3, 4, 19, 33,  0), # a = 0.03, f = 0.89, 45 deg. oblique
    datetime(2022, 3, 4, 19, 37,  0), # a = 0.03, f = 0.89, 45 deg. oblique
    datetime(2022, 3, 4, 19, 41,  0), # a = 0.06, f = 0.64, 45 deg. oblique

]

titles = [
    "Hs = 4 ft, Tp = 7 s, orthogonal",
    "Hs = 8 ft, Tp = 10 s, orthogonal",
    "Hs = 8 ft, Tp = 14 s, 10 deg. oblique",
    "Hs = 8 ft, Tp = 14 s, 10 deg. oblique",
    "Hs = 4 ft, Tp = 7 s, 30 deg. oblique",
    "Hs = 8 ft, Tp = 10 s, 30 deg. oblique",
    "Hs = 8 ft, Tp = 10 s, 30 deg. oblique",
    "Hs = 4 ft, Tp = 5 s, 45 deg. oblique",
    "Hs = 4 ft, Tp = 5 s, 45 deg. oblique",
    "Hs = 8 ft, Tp = 7 s, 45 deg. oblique"
]

filenames = [
    #'4ft_7s_baseline',
    #'8ft_10s_baseline',
    #'4ft_5s_baseline',
    #'8ft_7s_baseline',
    #'8ft_14s_baseline',
    '4ft_7s_orthogonal',
    '8ft_10s_orthogonal',
    '8ft_14s_10deg_a',
    '8ft_14s_10deg_b',
    '4ft_7s_30deg',
    '8ft_10s_30deg_a',
    '8ft_10s_30deg_b',
    '4ft_5s_45deg_a',
    '4ft_5s_45deg_b',
    '8ft_7s_45deg',
]

height = [#4, 8, 4, 8, 8,
    4, 8, 8, 8, 4, 8, 8, 4, 4, 8]
period = [#7, 10, 5, 7, 14,
    7, 10, 14, 14, 7, 10, 10, 5, 5, 7]

experiment = [
    #'baseline',
    #'baseline',
    #'baseline',
    #'baseline',
    #'baseline',
    'model_orthogonal',
    'model_orthogonal',
    'model_10deg',
    'model_10deg',
    'model_30deg',
    'model_30deg',
    'model_30deg',
    'model_45deg',
    'model_45deg',
    'model_45deg',
]


data_options = ['a1', 'a2', 'a3', 'v1', 'v2', 'v3', 'vtotal', 'c1', 'c2', 'c3']
units = ['counts', 'counts', 'counts', 'm/s', 'm/s', 'm/s', 'm/s', '%', '%', '%']

#connect the two aquadopp files
def findOtherAq(base1, base2):
    if (os.path.exists('datepickles/' + base1 + '.pickle') == False or os.path.exists('datepickles/' + base2 + '.pickle') == False):
        print('failed')
    else:
        file = open('datepickles/' + base1 + '.pickle', 'rb')
        data_dict1 = pickle.load(file)
        file = open('datepickles/' + base2 + '.pickle', 'rb')
        data_dict2 = pickle.load(file)

        data_dict1['find_other'] =  data_dict2['base_filename']
        data_dict2['find_other'] =  data_dict1['base_filename']
        
        saveAsPickle(data_dict1, base1)
        saveAsPickle(data_dict2, base2)
    return

#open pickle with dict
def openPickle(base_name):
    file = open('datepickles/' + base_name + '.pickle', 'rb')
    d = pickle.load(file)
    return d   

#add location in tank to pickle
def addOrder(base_name, order):
    file = open('datepickles/' + base_name + '.pickle', 'rb')
    d = pickle.load(file)
    d['location'] = order
    saveAsPickle(d,base_name)

 #save pickle, not neccesaary but makes opening dicts faster
def saveAsPickle(dictionary, base_name):
    pickle_out = open('datepickles/' + base_name +'.pickle', 'wb')
    pickle.dump(dictionary, pickle_out)
    pickle_out.close() 

#gets speed headings from csv file
def createSpeedHeadings(base_name):
    headings = []
    f = pd.read_csv(base_name + '.csv',delimiter = ';' )
    csv_headers = f.columns
    for i in range(9, len(csv_headers)-1, 2):
        headings.append(csv_headers[i])

    return headings, f

#creates dict of all necessary information and combined the files created from the PRF files
#saves at pickle that can be opened what want to use data
def createDict(base_name, location='still needed', find_other = 'still needed'):
    new_dict = {}
    DateTime_conv = []
    
    headings, csv = createSpeedHeadings(base_name)
    a1 = openASCIIFile(base_name + '.a1')
    a2 = openASCIIFile(base_name + '.a2')
    a3 = openASCIIFile(base_name + '.a3')
    c1 = openASCIIFile(base_name + '.c1')
    c2 = openASCIIFile(base_name + '.c2')
    c3 = openASCIIFile(base_name + '.c3')
    v1 = openASCIIFile(base_name + '.v1')
    v2 = openASCIIFile(base_name + '.v2')
    v3 = openASCIIFile(base_name + '.v3')
    f  = open(base_name + '.hdr', 'r')
    hdr = f.read() 
    f.close()
    base_name = base_name.split('/')[-1]
    print(base_name)
    new_dict['cell_size'] = float(hdr.split("Cell size")[1].split('mm', 1)[0])/100
    new_dict['cell_number'] = int(hdr.split("Number of cells")[1].split('A', 1)[0])
    new_dict['base_name'] = base_name  #KRD change this to be consistent
    new_dict['date'] = hdr.split('Time of first measurement')[1].split('PM')[0]
    new_dict['headings'] = headings
    new_dict['location'] = location
    for x in csv['DateTime']:
        DateTime_conv.append(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
    DateTime_conv = np.array(DateTime_conv)
    new_dict['datetime'] = DateTime_conv
    new_dict['find_other'] = find_other
    
    for i in range(0, len(headings)):
        depth_id = headings[i].split('(')[1].split(')')[0] 
        new_dict[headings[i]] = {'a1': a1[:, i+2], 'a2': a2[:, i+2], 'a3': a3[:, i+2], 
                              'c1': c1[:, i+2], 'c2': c2[:, i+2], 'c3': c3[:, i+2],
                              'v1': v1[:, i+2], 'v2': v2[:, i+2], 'v3': v3[:, i+2],
                              'avg_speed': csv[headings[i]], 'dtoADCP': round((float(depth_id.split('m')[0])),3) } #KRD 0.145 is distance for katies experiemnt (need to change)
                              #need to add distnace from aquadopp to bottom                   
    new_dict = velocityTotal(new_dict) 
    saveAsPickle(new_dict, base_name)   
    return new_dict

#computes magnitude of total velocity    
def velocityTotal(data_dict):
    headings = data_dict['headings']
    for i in range(0, len(headings)):
        velocity_total = []
        for j in range(0, len(data_dict[headings[i]]['v3'])):
            
            velocity_total.append(np.sqrt((data_dict[headings[i]]['v1'][j]**2 + 
                        data_dict[headings[i]]['v2'][j]**2+
                        data_dict[headings[i]]['v3'][j]**2)))
        data_dict[headings[i]]['vtotal'] = velocity_total
    return data_dict
    
#not currently called anywhere
#computes signal to noise ratio
def signalToNoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

#plot signal to noise but not currently used
def plotSigtoNoise(base_filename, title, characteristic):  #TODO update with working dict
    data = createDict(base_filename)
    signoise = []
    for num in data['headings']:
        signoise.append(signalToNoise(np.asarray(data[num][characteristic])))
        
    plt.plot(SpeedNumbers, signoise, label = data['location'])
    plt.xticks(rotation = 90)
    plt.xlabel('vertical location')
    plt.ylabel('signal to noise ratio')
    plt.legend()
    plt.title(title)

#convert string to datetime
def toDateTime(data):
    data['DateTimeConv'] = data['DateTime']
    for i in range(0,len(data['DateTime'])):
        data['DateTimeConv'][i] = datetime.strptime(data['DateTime'][i], "%m/%d/%Y %H:%M:%S")
    print(type(data['DateTimeConv'][0]))
    return data

#open file and convert date to array
def openASCIIFile(filename):
    
    f = open(filename, 'r')
    data = np.genfromtxt(f)
    f.close()
    return data

#determine where data starts and end with given conditions
def sectionData(data_dict, start):
    DateTime = data_dict['datetime']
    abs_deltas_from_target_date = np.absolute(DateTime - start)
    index_of_start = np.argmin(abs_deltas_from_target_date)
    end = start+timedelta(seconds=30)
    abs_deltas_from_target_date = np.absolute(DateTime - end)
    index_of_end = np.argmin(abs_deltas_from_target_date)

    
    return index_of_start, index_of_end

def meanProfileReefline(uptank, cond = 'still_needed', filtering = 'on'):
    for i in range(0,len(runs_start)):
        #plotConditionsAquadoppReefline(i, uptank, cond, filtering, runs_start[i])
        upavg, downavg, upvar, downvar = averageData(i, uptank, runs_start[i])
        plotProfileReefline(upavg, downavg, upvar, downvar, i, runs_start[i])
    return

def plotProfileReefline(upavg, downavg, upvar, downvar, trial_num, runs_start):
    fig, ax = plt.subplots(1,3,  figsize =(15,12))
    units = ['m/s', '%', 'm/s']
    labels = ['velocity alongtank', 'correlation along tank', 'total velocity']
    for i in range(1,4):
        ax[i-1].plot(upavg[:,i], upavg[:,0], label = 'uptank from reef', color = 'green')
        ax[i-1].plot(downavg[:,i], downavg[:,0], label = 'downtank from reef', color = 'darkorange')
        if (i==1 or i==3):
            ax[i-1].errorbar(upavg[:,i], upavg[:,0], xerr = upvar[:,i],  color = 'green', alpha=0.7, 
                    fmt='o', ecolor='mediumseagreen', elinewidth=3, capsize=0)
            ax[i-1].errorbar(downavg[:,i], downavg[:,0], xerr = downvar[:,i], color = 'darkorange', alpha = 0.6, 
                fmt='o', ecolor='sandybrown', elinewidth=3, capsize=0)
        ax[i-1].set_ylabel('distance to bottom (m)')
        ax[i-1].set_xlabel(units[i-1])
        ax[i-1].set_title(labels[i-1])
        ax[i-1].legend()
    fig.suptitle(runs_start, fontsize = 20)
    # fig.savefig('profiles/' + str(cond['realnum'][trial_num]) +'_'+ str(cond['start'][trial_num]), dpi=100)
    # plt.close('all')
    return

def plotConditionsAquadoppReefline(i, base, cond, filtering, runs_start):
    alpha = 1
    desired_dict = 'vtotal'  
    first_bin = 0 #TODO dont hard code this
    downtank_dict = openPickle(base)
    uptank_dict = openPickle(downtank_dict['find_other'])
    if(filtering == 'on'): 
        alpha = 0.2
    headingup = uptank_dict['headings']
    headingdown = downtank_dict['headings']
    fig, ax = plt.subplots(3,2,  figsize =(20,30))
    count = 0
    for j in range (0,3):
        for k in range (0,2):
            start_u, end_u = sectionData(uptank_dict, runs_start)
            ax[j][k].plot(uptank_dict['datetime'][start_u:end_u], 
                            uptank_dict[headingup[first_bin+count]][desired_dict][start_u:end_u], 
                            label = uptank_dict['location'], alpha = alpha, color = 'green')
            start_d, end_d = sectionData(downtank_dict, runs_start)
            ax[j][k].plot(downtank_dict['datetime'][start_d:end_d], 
                            downtank_dict[headingdown[first_bin+count]][desired_dict][start_d:end_d], 
                            label = downtank_dict['location'], alpha = alpha, color='orange')
            
            ax[j][k].set_title("conditions needed", fontsize = 20)
            ax[j][k].tick_params(axis = 'x', labelrotation=45, labelsize = 16)
            ax[j][k].tick_params(axis = 'y', labelsize = 16)
            ax[j][k].set_ylabel(str(downtank_dict[headingdown[first_bin+count]]['dtoADCP']) + ' m' + '\n m/s', fontsize = 20)
            if (filtering == 'on'):
                upfilter, downfilter = filterData(uptank_dict[headingup[first_bin+count]][desired_dict][start_u:end_u],
                                                    downtank_dict[headingdown[first_bin+count]][desired_dict][start_d:end_d])                                   
                ax[j][k].plot(uptank_dict['datetime'][start_u:end_u], 
                            upfilter, label = uptank_dict['location'] + ' filtered', color = 'green')
                ax[j][k].plot(downtank_dict['datetime'][start_d:end_d], 
                            downfilter, label = downtank_dict['location'] + ' filtered', color='orange')
            ax[j][k].legend(fontsize = 14)
            count+=1
    fig.suptitle(runs_start, y=0.9, fontsize = 24)


#read conditions file 
#would need to be updated for a differently formatted 
#       conditions file this is based on Katie's conditions file
"""
def readConditions(cond_file, base_name):
    #base_name = base_name.split('/')[1] + '.PRF'
    names = f['base_name']  
    names = np.asarray(names)
    found_start = False
    first = 0
    last = 0
    for i in range(0,len(names)):
        if (base_name == names[i] and found_start==False):
            first = i
            found_start = True
        if (base_name != names[i] and found_start):
            last = i
            break
    for i in range(first,last):
        if(isinstance(start_arr[i], float)):
            start[i] = np.nan
            end[i] = np.nan
            realstart[i] = np.nan
            continue
        total_datetime = date[i] + ' ' + start_arr[i]
        total_datetime_real = date[i] + ' ' + realstart_arr[i]
        
        start[i] = datetime.strptime(total_datetime, "%m/%d/%Y %H:%M")
        end[i] = start[i]+timedelta(seconds=(time_diff_arr[i]*60))
        realstart[i] = datetime.strptime(total_datetime_real, "%m/%d/%Y %H:%M:%S")
    start = np.array(start)
    end = np.array(end)
    realstart = np.array(realstart)
    cond_dict = {'names': names, 'trial_num': trial_num, 'coral': coral, 'period': period, 
                    'freq': freq, 'h_water': h_water, 'h_wave': h_wave, 'amp': amp, 
                    'wind': wind, 'start': start, 'end': end, 'realnum': realnum, 'realstart': realstart}
    return cond_dict
"""








#plots the averaged filtered data in all bins with error bars based on unfiltered data
#plots v1 (along tank for katie's experiment) and vtotal as well as correlation c1
def meanProfile(upavg, downavg, upvar, downvar, cond, trial_num, base):
    fig, ax = plt.subplots(1,3,  figsize =(15,12))
    units = ['m/s', '%', 'm/s']
    labels = ['velocity alongtank', 'correlation along tank', 'total velocity']
    for i in range(1,4):
        ax[i-1].plot(upavg[:,i], upavg[:,0], label = 'uptank from reef', color = 'green')
        ax[i-1].plot(downavg[:,i], downavg[:,0], label = 'downtank from reef', color = 'darkorange')
        if (i==1 or i==3):
            ax[i-1].errorbar(upavg[:,i], upavg[:,0], xerr = upvar[:,i],  color = 'green', alpha=0.7, 
                    fmt='o', ecolor='mediumseagreen', elinewidth=3, capsize=0)
            ax[i-1].errorbar(downavg[:,i], downavg[:,0], xerr = downvar[:,i], color = 'darkorange', alpha = 0.6, 
                fmt='o', ecolor='sandybrown', elinewidth=3, capsize=0)
        ax[i-1].set_ylabel('distance to bottom (m)')
        ax[i-1].set_xlabel(units[i-1])
        ax[i-1].set_title(labels[i-1])
        ax[i-1].legend()
    fig.suptitle(str(cond['start'][i]) + "\nTs=" + str(cond['freq'][trial_num]) + 
                " s  h=" + str(cond['h_wave'][trial_num]) +
                " m  water height=" + str(cond['h_water'][trial_num]) + " m\n Coral Configuration: " + str(cond['coral'][trial_num]), fontsize = 20)
    fig.savefig('profiles/' + str(cond['realnum'][trial_num]) +'_'+ str(cond['start'][trial_num]), dpi=100)
    plt.close('all')
    return

#plots wave condtions in the order of increasing period and wave height
#plots same information as mean profile but in a different order
#can be changed for data you wnat to plot
def conditonsGradient(upavg, downavg, upvar, downvar, cond, base_name, entire_dict):  #entire dict to plot all configurations together
    data_want = 3   #1=v1, 2=c1, 3=vtotal
    fig, ax = plt.subplots(3,3,  figsize =(20,30))
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="w", ec="k", lw=2)
    ax[0][0].annotate('                                increasing wave height                             ',
            xy=(400, 2800), xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props, fontsize = 30)
    count = 0
    bbox_props = dict(boxstyle="larrow,pad=0.5", fc="w", ec="k", lw=2)
    ax[0][0].annotate('                                          decreasing period                                      ',
            xy=(80, 2000), xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props, rotation = 90, fontsize = 30)
    count = 0
    period_want = [0.81, 0.81, 0.81, 0.63, 0.63, 0.63, 0.4, 0.4, 0.4]  #this is hard coded for Katies experiment
    ts = 0
    #this is putting all data from all experiemtns in one dictionary to plot comparison of coral configurations
    #not necessary for all experiments
    entire_dict[base_name]={}
    entire_dict[base_name]['coral'] = str(cond['coral'][0])
    entire_dict[base_name]['h_water'] = str(cond['h_water'][0])
    entire_dict[base_name]['depth'] = upavg[str(count)][:,0] 
    entire_dict[base_name]['difference'] = {}
    for j in range(0,3):
        for i in range(0, 3):
            while(period_want[ts]!=cond['freq'][count]):
                count +=1
            ax[j][i].plot(upavg[str(count)][:,data_want], upavg[str(count)][:,0], label = 'uptank from reef', color = 'green')
            ax[j][i].plot(downavg[str(count)][:,data_want], downavg[str(count)][:,0], label = 'downtank from reef', color = 'darkorange')
            ax[j][i].set_xlabel('m/s', fontsize = 20)
            ax[j][i].set_title("Ts=" + str(cond['freq'][count]) + " s  h=" + str(cond['h_wave'][count]) + " m", fontsize = 20)
            ax[j][i].errorbar(upavg[str(count)][:,data_want], upavg[str(count)][:,0], xerr = upvar[str(count)][:,data_want],  color = 'green', alpha=0.7, 
                    fmt='o', ecolor='mediumseagreen', elinewidth=3, capsize=0)
            ax[j][i].errorbar(downavg[str(count)][:,data_want], downavg[str(count)][:,0], xerr = downvar[str(count)][:,data_want], color = 'darkorange', alpha = 0.6, 
                fmt='o', ecolor='sandybrown', elinewidth=3, capsize=0)
            ax[j][i].tick_params(axis = 'x', labelsize = 13)
            ax[j][i].tick_params(axis = 'y', labelsize = 13)
            ax[j][i].legend()
            entire_dict[base_name]['difference'][ts] = upavg[str(count)][:,data_want]-downavg[str(count)][:,data_want]
            count +=1
            ts +=1
    ax[1][0].set_ylabel('distance to bottom (m)', fontsize = 20) 
        #ax[d][0].legend(fontsize = 12)  
    fig.suptitle(str(cond['start'][i]).split()[0] + "  water height=" + str(cond['h_water'][0]) + " m\n Coral Configuration: " + str(cond['coral'][0]), fontsize = 40)
    fig.savefig('comparetwo/' + base_name, dpi=100)
    plt.close('all')
    return entire_dict

#plots all conditions for a single experiemnt in one subplot
def compareConditions(upavg, downavg, upvar, downvar, cond, base_name):
    d = 0
    units = ['m/s', '%', 'm/s']
    fig, ax = plt.subplots(3,len(cond['realnum']),  figsize =(60,30))
    for data_wanted in ('v1', 'c1', 'vtotal'):
        for i in range(0, len(cond['realnum'])):
            ax[d][i].plot(upavg[str(i)][:,d+1], upavg[str(i)][:,0], label = 'uptank from reef', color = 'green')
            ax[d][i].plot(downavg[str(i)][:,d+1], downavg[str(i)][:,0], label = 'downtank from reef', color = 'darkorange')
            ax[d][i].set_xlabel(units[d], fontsize = 20)
            ax[d][i].set_title("Ts=" + str(cond['freq'][i]) + " s  h=" + str(cond['h_wave'][i]) + " m", fontsize = 20)
            if (d==0 or d==2):
                ax[d][i].errorbar(upavg[str(i)][:,d+1], upavg[str(i)][:,0], xerr = upvar[str(i)][:,d+1],  color = 'green', alpha=0.7, 
                        fmt='o', ecolor='mediumseagreen', elinewidth=3, capsize=0)
                ax[d][i].errorbar(downavg[str(i)][:,d+1], downavg[str(i)][:,0], xerr = downvar[str(i)][:,d+1], color = 'darkorange', alpha = 0.6, 
                    fmt='o', ecolor='sandybrown', elinewidth=3, capsize=0)
            ax[d][i].tick_params(axis = 'x', labelsize = 13)
            ax[d][i].tick_params(axis = 'y', labelsize = 13)
        ax[d][0].set_ylabel(data_wanted + ' \ndistance to bottom (m)', fontsize = 20) 
        #ax[d][0].legend(fontsize = 12)
        d+=1   
    fig.suptitle(str(cond['start'][i]).split()[0] + "  water height=" + str(cond['h_water'][0]) + " m\n Coral Configuration: " + str(cond['coral'][0]), fontsize = 40)
    fig.savefig('profiles/' + base_name, dpi=100)
    plt.close('all')
    return

#overarching plotting function for all types of plots
#also can be toggled to filter data or not
def conditionsSlider(conditions, base_name, filtering='off', entire_dict={}):
    
    cond_dict = readConditions(conditions, base_name)
    # interact(plotConditionsAquadopp, i = (0,len(cond_dict['trial_num'])), data_want = (0,len(data_options)),
    #         base = fixed(base_name), cond = fixed(cond_dict), filtering = fixed(filtering))
    up_all_trials = {}
    down_all_trials = {}
    upvar_all_trials = {}
    downvar_all_trials = {}
    for i in range(0,len(cond_dict['trial_num'])):
        upavg, downavg, upvar, downvar = averageData(i, base_name, cond_dict)
        up_all_trials[str(i)] = upavg
        down_all_trials[str(i)] = downavg
        upvar_all_trials[str(i)] = upvar
        downvar_all_trials[str(i)] = downvar
        #meanProfile(upavg, downavg, upvar, downvar, cond_dict, i, base_name)
        # for data_want in (3,6):
        #     plotConditionsAquadopp(i, data_want, base_name, cond_dict, filtering)
    entire_dict = conditonsGradient(up_all_trials, down_all_trials, upvar_all_trials, 
                                    downvar_all_trials, cond_dict, base_name, entire_dict)
    #compareConditions(up_all_trials, down_all_trials, upvar_all_trials, downvar_all_trials, cond_dict, base_name)
    return entire_dict


#averages filtered data and finds variance of unfiltered data
def averageData(num, base, start_times):
    downtank_dict = openPickle(base)
    uptank_dict = openPickle(downtank_dict['find_other'])
    headingup = uptank_dict['headings']
    
    headingdown = downtank_dict['headings']
    
    upfilter = []
    upvar = []
    downfilter = []
    downvar = []
  
    for i in range (0,downtank_dict['cell_number']):  
        start_u, end_u = sectionData(uptank_dict, start_times)
        #start_ureal, end_ureal = sectionData(uptank_dict, start_times)
        start_d, end_d = sectionData(downtank_dict, start_times)
        #start_dreal, end_dreal = sectionData(downtank_dict, start_times)
        dist_up = float(headingup[i].split('(')[1].split(')')[0].split('m')[0])
        if(dist_up >= water_depth):
            break
        upfilter.append(dist_up)
        upvar.append(dist_up)
        dist_down = float(headingdown[i].split('(')[1].split(')')[0].split('m')[0])
        downfilter.append(dist_down)
        downvar.append(dist_down)
        
        for data in ('v1', 'c1', 'vtotal'): #KRD this is based on katies experiment 
            up, down = filterData(uptank_dict[headingup[i]][data][start_u:end_u],
                                                        downtank_dict[headingdown[i]][data][start_d:end_d])
            upfilter.append(round(np.mean(up),3))
            downfilter.append(round(np.mean(down),3))
            upvar.append(np.var(uptank_dict[headingup[i]][data][start_u:end_u]))
            downvar.append(np.var(downtank_dict[headingdown[i]][data][start_d:end_d]))
        
       
    downfilter = np.reshape(downfilter, (i,4))
    upfilter = np.reshape(upfilter, (i,4))
    downvar = np.reshape(downvar, (i,4))
    upvar = np.reshape(upvar, (i,4))
    return upfilter, downfilter, upvar, downvar

#plots each bin individually and can be filtered or not 
#for one condition
#based on Katies condition file might need to change
def plotConditionsAquadopp(i, data_want, base, cond, filtering):
    alpha = 1
    desired_dict = data_options[data_want]  
    first_bin = 0 #TODO dont hard code this
    downtank_dict = openPickle(base)
    uptank_dict = openPickle(downtank_dict['find_other'])
    if(filtering == 'on'): 
        alpha = 0.2
    headingup = uptank_dict['headings']
    headingdown = downtank_dict['headings']
    fig, ax = plt.subplots(3,2,  figsize =(20,30))
    count = 0
    for j in range (0,3):
        for k in range (0,2):
            if(isinstance(cond['start'][i], float)):
                continue

            start_u, end_u = sectionData(uptank_dict, cond['start'][i], cond['end'][i])
            ax[j][k].plot(uptank_dict['datetime'][start_u:end_u], 
                            uptank_dict[headingup[first_bin+count]][desired_dict][start_u:end_u], 
                            label = uptank_dict['location'], alpha = alpha, color = 'green')
            start_d, end_d = sectionData(downtank_dict, cond['start'][i], cond['end'][i])
            ax[j][k].plot(downtank_dict['datetime'][start_d:end_d], 
                            downtank_dict[headingdown[first_bin+count]][desired_dict][start_d:end_d], 
                            label = downtank_dict['location'], alpha = alpha, color='orange')
            
            ax[j][k].set_title("Ts=" + str(cond['freq'][i]) + " h=" + str(cond['h_wave'][i]), fontsize = 20)
            ax[j][k].tick_params(axis = 'x', labelrotation=45, labelsize = 16)
            ax[j][k].tick_params(axis = 'y', labelsize = 16)
            ax[j][k].set_ylabel(str(downtank_dict[headingdown[first_bin+count]]['dtoADCP']) + ' m' + '\n ' + units[data_want], fontsize = 20)
            if (filtering == 'on'):
                upfilter, downfilter = filterData(uptank_dict[headingup[first_bin+count]][desired_dict][start_u:end_u],
                                                    downtank_dict[headingdown[first_bin+count]][desired_dict][start_d:end_d])
                ax[j][k].plot(uptank_dict['datetime'][start_u:end_u], 
                            upfilter, label = uptank_dict['location'] + ' filtered', color = 'green')
                ax[j][k].plot(downtank_dict['datetime'][start_d:end_d], 
                            downfilter, label = downtank_dict['location'] + ' filtered', color='orange')
            ax[j][k].legend(fontsize = 14)
            count+=1
    fig.suptitle(uptank_dict['date'] + ' ' + desired_dict, y=0.9, fontsize = 24)
    fig.savefig('plots/' + base + str(cond['start'][i]) + desired_dict + '_' + str(cond['realnum'][i]), dpi=100)
    plt.close('all')

#filter data using convolution
def filterData(up,down):
    box_pts = 5
    box = np.ones(box_pts)/box_pts
    upfilter = np.convolve(up, box, mode = 'same')
    downfilter = np.convolve(down, box, mode = 'same')
    return upfilter, downfilter 

#plots all coral configurations together
#the difference between uptank and downtank
def plotAllConfig(entire_dict):
    period = [0.81, 0.81, 0.81, 0.63, 0.63, 0.63, 0.4, 0.4, 0.4]
    height = [0.08, 0.1, 0.12, 0.08, 0.1, 0.12, 0.08, 0.1, 0.12] 
    working_files = [ 'SEP2020', 'SEP2322', 'OCT428', 'OCT529', 'OCT803', 'OCT1910', 'OCT1911', 'OCT2012', 'OCT2113']
    fig, ax = plt.subplots(3,3,  figsize =(20,30))
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="w", ec="k", lw=2)
    ax[0][0].annotate('                                increasing wave height                             ',
            xy=(400, 2800), xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props, fontsize = 30)
   
    bbox_props = dict(boxstyle="larrow,pad=0.5", fc="w", ec="k", lw=2)
    ax[0][0].annotate('                                          decreasing period                                      ',
            xy=(80, 2000), xycoords='figure pixels', horizontalalignment='left', verticalalignment='top', bbox=bbox_props, rotation = 90, fontsize = 30)
    for file in working_files:
        count = 0
        file = 'comm4/' + file
        for j in range(0,3):
            for i in range(0, 3):
                ax[j][i].plot(entire_dict[file]['difference'][count], entire_dict[file]['depth'],  label =  entire_dict[file]['coral'])
                ax[j][i].set_xlabel('m/s', fontsize = 20)
                ax[j][i].set_title("Ts=" + str(period[count]) + " s  h=" + str(height[count]) + " m", fontsize = 20)
                ax[j][i].tick_params(axis = 'x', labelsize = 13)
                ax[j][i].tick_params(axis = 'y', labelsize = 13)
                ax[j][i].legend()
                ax[j][i].set_xlim([-0.1, 0.17])
                count+=1
    ax[1][0].set_ylabel('distance to bottom (m)', fontsize = 20) 
    fig.suptitle("Comparison of all coral configurations", fontsize = 30)
    fig.savefig("comparison_of_all_configs", dpi = 100)
    plt.close('all')
    return 




#can be used to create table with data
   # print(headingup)
    # print(headingdown)
    # x = PrettyTable()
    # x.field_names = ['Loc','Trial#', 'water ht', 'Frequency', 'wave ht',
    #                      'depth', 'avg v2', 'avg vtotal', 'avg c2']
        # x.add_row([uptank_dict['location'], num, cond['h_water'][num], cond['freq'][num], 
        #             cond['h_wave'][num], str(round(dist_up,3)) + ' m', 
        #             upfilter[0], upfilter[1], upfilter[2]])
        # x.add_row([downtank_dict['location'], ' ', ' ', ' ', ' ', 
        #             str(round(dist_down,3)) + ' m', 
        #             downfilter[0], downfilter[1], downfilter[2]])
        # if(downfilter[0] > upfilter[0]):
        #     bigger_v2 = 'downtank'
        # else:
        #      bigger_v2 = 'uptank'
        # if(downfilter[1] > upfilter[1]):
        #     bigger_vtotal = 'downtank'
        # else:
        #     bigger_vtotal = 'uptank'
    #     x.add_row([' ',' ', ' ', ' ', ' ', ' ', bigger_v2, bigger_vtotal, ' '])
    # print(x)
    