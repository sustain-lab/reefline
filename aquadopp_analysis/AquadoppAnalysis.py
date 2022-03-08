import string
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pickle

fileBase4 = ['OCT803', 'OCT702', 'OCT126', 'OCT127', 'OCT428', 'OCT529', 'OCT530', 'OCT631', 'SEP _315', 
             'SEP316', 'SEP317', 'SEP318', 'SEP319', 'SEP2020', 'SEP2221', 'SEP2322 revise', 'SEP2322', 
             'SEP2823', 'SEP2924', 'SEP3025', 'OCT1204', 'OCT1305', 'OCT1306', 'OCT1507', 'OCT1508', 
             'OCT15309', 'OCT1910', 'OCT1911', 'OCT2012', 'OCT2113', 'OCT2214', 'NOV515', 'NOV916', 
             'NOV1717', 'NOV1718', 'NOV2219', 'NOV2320', 'NOV2921', 'NOV3022', 'DEC123', 'DEC124', 
             'DEC325', 'DEC626', 'DEC1027', 'DEC1028', 'DEC1229']
fileBase5 = ['OCT829', 'OCT728', 'OCT627', 'OCT526', 'OCT525', 'OCT424', 'OCT123', 'OCT122', 'SEP3021', 
             'SEP2920', 'SEP2819', 'SEP2318', 'SEP2217', 'SEP2016', 'SEP37215', 'SEP313', 'SEP31212', 
             'OCT15203', 'OCT15304', 'OCT1905', 'OCT1906', 'OCT2007', 'OCT2108', 'OCT2209', 'NOV510', 
             'NOV911', 'NOV1712', 'NOV1713', 'NOV2214', 'NOV2316', 'NOV2917', 'NOV3018', 'DEC119', 
             'DEC120', 'DEC321', 'DEC622']

#def findOtherAq(baseName):
    

def addOrder(baseName, order):
    file = open('datepickles/' + baseName + '.pickle', 'rb')
    d = pickle.load(file)
    d['location'] = order
    saveAsPickle(d,)

def saveAsPickle(dictionary, baseName):
    pickle_out = open('datepickles/' + baseName +'.pickle', 'wb')
    pickle.dump(dictionary, pickle_out)
    pickle_out.close() 

def createSpeedHeadings(base_filename):
    headings = []
    f = pd.read_csv(base_filename + '.csv',delimiter = ';' )
    csv_headers = f.columns
    for i in range(9, len(csv_headers)-1, 2):
        headings.append(csv_headers[i])

    return headings, f

def create_dict(base_filename, location='still needed'):
    new_dict = {}
    DateTime_conv = []

    headings, csv = createSpeedHeadings(base_filename)
    a1 = OpenASCIIFile(base_filename + '.a1')
    a2 = OpenASCIIFile(base_filename + '.a2')
    a3 = OpenASCIIFile(base_filename + '.a3')
    c1 = OpenASCIIFile(base_filename + '.c1')
    c2 = OpenASCIIFile(base_filename + '.c2')
    c3 = OpenASCIIFile(base_filename + '.c3')
    v1 = OpenASCIIFile(base_filename + '.v1')
    v2 = OpenASCIIFile(base_filename + '.v2')
    v3 = OpenASCIIFile(base_filename + '.v3')
    f  = open(base_filename + '.hdr', 'r')
    hdr = f.read() 
    f.close()
    new_dict['cell_size'] = float(hdr.split("Cell size")[1].split('mm', 1)[0])/100
    new_dict['cell_number'] = int(hdr.split("Number of cells")[1].split('A', 1)[0])
    new_dict['base_filename'] = base_filename
    new_dict['date'] = hdr.split('Time of first measurement')[1].split('PM')[0]
    new_dict['headings'] = headings
    new_dict['location'] = location
    for x in csv['DateTime']:
        DateTime_conv.append(datetime.strptime(x, "%m/%d/%Y %H:%M:%S"))
    DateTime_conv = np.array(DateTime_conv)
    new_dict['DateTime'] = DateTime_conv

    for i in range(0, len(headings)-1):
        depth_id = headings[i].split('(')[1].split(')')[0] 
        new_dict[headings[i]] = {'a1': a1[:, i+2], 'a2': a2[:, i+2], 'a3': a3[:, i+2], 
                              'c1': c1[:, i+2], 'c2': c2[:, i+2], 'c3': c3[:, i+2],
                              'v1': v1[:, i+2], 'v2': v2[:, i+2], 'v3': v3[:, i+2],
                              'avg_speed': csv[headings[i]], 'dtoADCP': float(depth_id.split('m')[0]) }
        
    return new_dict
    
    

    


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def plotSigtoNoise(base_filename, title, characteristic):  #TODO update with working dict
    data = create_dict(base_filename)
    signoise = []

    for num in data['headings']:
        signoise.append(signaltonoise(np.asarray(data[num][characteristic])))
        
    plt.plot(SpeedNumbers, signoise, label = data['location'])
    plt.xticks(rotation = 90)
    plt.xlabel('vertical location')
    plt.ylabel('signal to noise ratio')
    plt.legend()
    plt.title(title)
    #plt.show() 

def toDateTime(data):
    data['DateTimeConv'] = data['DateTime']
    for i in range(0,len(data['DateTime'])):
        data['DateTimeConv'][i] = datetime.strptime(data['DateTime'][i], "%m/%d/%Y %H:%M:%S")
    print(type(data['DateTimeConv'][0]))
    return data


def PlotExperiment(comm4, comm5, waveconditions):
    fig1, ax1 = plt.subplots(1, 4, figsize =(20,5))
    fig2, ax2 = plt.subplots(1, 3, figsize =(15,5))
    
    fig1.suptitle(waveconditions, fontsize = 16)
    
    ax1[0].plot(comm4['DateTimeConv'], comm4['Speed#1(0.150m)'], alpha = 0.5)
    ax1[0].plot(comm5['DateTimeConv'], comm5['Speed#1(0.150m)'], 'r-', alpha = 0.5)
    ax1[0].tick_params(labelrotation=45)
    ax1[0].set_title('Speed #1 (0.150m)')
    ax1[0].set_xlabel('time')
    ax1[0].set_ylabel('speed')

    ax1[1].plot(comm4['DateTimeConv'], comm4['Speed#2(0.200m)'], alpha = 0.5)
    ax1[1].plot(comm5['DateTimeConv'], comm5['Speed#2(0.200m)'], 'r-', alpha = 0.5)
    ax1[1].tick_params(labelrotation=45)
    ax1[1].set_title('Speed #2 (0.200m)')
    ax1[1].set_xlabel('time')
    ax1[1].set_ylabel('speed')
    
    ax1[2].plot(comm4['DateTimeConv'], comm4['Speed#3(0.250m)'], alpha = 0.5)
    ax1[2].plot(comm5['DateTimeConv'], comm5['Speed#3(0.250m)'], 'r-', alpha = 0.5)
    ax1[2].tick_params(labelrotation=45)
    ax1[2].set_title('Speed #3 (0.250m)')
    ax1[2].set_xlabel('time')
    ax1[2].set_ylabel('speed')

    ax1[3].plot(comm4['DateTimeConv'], comm4['Speed#4(0.300m)'], alpha = 0.5)
    ax1[3].plot(comm5['DateTimeConv'], comm5['Speed#4(0.300m)'], 'r-', alpha = 0.5)
    ax1[3].tick_params(labelrotation=45)
    ax1[3].set_title('Speed #4 (0.300m)')
    ax1[3].set_xlabel('time')
    ax1[3].set_ylabel('speed')

    ax2[0].plot(comm4['DateTimeConv'], comm4['Speed#5(0.350m)'], alpha = 0.5)
    ax2[0].plot(comm5['DateTimeConv'], comm5['Speed#5(0.350m)'], 'r-', alpha = 0.5)
    ax2[0].tick_params(labelrotation=45)
    ax2[0].set_title('Speed #5 (0.350m)')
    ax2[0].set_xlabel('time')
    ax2[0].set_ylabel('speed')

    ax2[1].plot(comm4['DateTimeConv'], comm4['Speed#6(0.400m)'], alpha = 0.5)
    ax2[1].plot(comm5['DateTimeConv'], comm5['Speed#6(0.400m)'], 'r-', alpha = 0.5)
    ax2[1].tick_params(labelrotation=45)
    ax2[1].set_title('Speed #6 (0.400m)')
    ax2[1].set_xlabel('time')
    ax2[1].set_ylabel('speed')

    ax2[2].plot(comm4['DateTimeConv'], comm4['Speed#7(0.450m)'], alpha = 0.5, label = 'comm4')
    ax2[2].plot(comm5['DateTimeConv'], comm5['Speed#7(0.450m)'], 'r-', alpha = 0.5, label = 'comm5')
    ax2[2].tick_params(labelrotation=45)
    ax2[2].legend(bbox_to_anchor=(1, 1))
    ax2[2].set_title('Speed #7 (0.450m)')
    ax2[2].set_xlabel('time')
    ax2[2].set_ylabel('speed')


def OpenASCIIFile(filename):
    
    f = open(filename, 'r')
    data = np.genfromtxt(f)
    f.close()
    #print(data.shape)

    return data

def PlotAmplitude(data_dict):
    plt.plot(sep319_csv['DateTimeConv'][2000:2100],sep319_a1[2000:2100,6])
    plt.ylabel('amplitude 1')
    plt.xlabel('time')
    plt.show()
    plt.plot(sep319_csv['DateTimeConv'][2000:2100],sep319_csv['Speed#4(0.300m)'][2000:2100])
    #plt.plot(sep319_csv['DateTimeConv'],sep319_c1[:,3])
    plt.ylabel('correlation 1')
    plt.xlabel('time')

    

# f = open(base_filename + '.hdr', 'r')
    # data = f.read()
    # f.close()
    # cell_size = float(data.split("Cell size")[1].split('mm', 1)[0])/100
    # cell_num = int(data.split("Number of cells")[1].split('A', 1)[0])

def SectionData(data_dict, start, end):
    
    DateTime = data_dict['DateTime']
    
    #whole_data = data_wanted
    # start_index=min(DateTime_conv, key=lambda x: abs(x - start))
    # end_index=min(DateTime_conv, key=lambda x: abs(x - end))
    abs_deltas_from_target_date = np.absolute(DateTime - start)
    index_of_start = np.argmin(abs_deltas_from_target_date)

    abs_deltas_from_target_date = np.absolute(DateTime - end)
    index_of_end = np.argmin(abs_deltas_from_target_date)
    # closest_date = all_dates[index_of_min_delta_from_target_date]

    return index_of_start, index_of_end

def PlotSection (uptank_dict, downtank_dict, desired_feature, bin_num, start = 0, run_time='nothing'):
    
    #feature = []
    for data in [uptank_dict, downtank_dict]:
        if (start!=0):
            heading = data['headings']
            end = start+timedelta(minutes=run_time)
            i_start, i_end = SectionData(data, start, end)
            print(data['DateTime'][i_end])
            feature = data[heading[bin_num]][desired_feature][i_start:i_end]
            plt.plot(data['DateTime'][i_start:i_end],feature/np.linalg.norm(feature), label = data['location'])
            plt.tick_params(axis = 'x', labelrotation=45)
            #plt.locator_params(axis = 'x', tight=True, nbins=30)
        else:
            feature.append(data[heading[bin_num]][desired_feature])
    plt.title(uptank_dict['date'])
    plt.xlabel('time of measurement')
    plt.ylabel('normalized ' + desired_feature)
    plt.legend()
    #plt.locator_params(numticks=12)
    #plt.locator_params(axis="x", nbins=4)
    # plt.tick_params(axis = 'x', labelrotation=90)
    # plt.locator_params(axis = 'x', tight=True, nbins=30)
    # plt.plot(uptank_dict['DateTime'],feature[0])
    # plt.plot(downtank_dict['DateTime'],feature[1])


"""
TODO:
1) Figure out dict, not sure its in the right order
2) Plot section function
3) Compartison function to find other file at same time and tag them as the same
4) Make pickle of things
* 5) Velocity componets to total velocity
6) Tag section to wave conditions and plot with it
* 7) Figure out bins for thesis and find ones that are the same
* 8) Calibration tests were deleted --> run again? 
9) Comment script
10) Upload to github and clone into sustain repo
""" 





    