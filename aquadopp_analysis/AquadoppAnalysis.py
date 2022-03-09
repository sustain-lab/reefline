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

# def findOtherAq(data_dict1, data_dict2):
#     data_dict1['find_other'] = 
    


#     return

    

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
    new_dict['datetime'] = DateTime_conv
    #print(len(DateTime_conv))
    new_dict['find_other'] = 'still needed'

    for i in range(0, len(headings)-1):
        depth_id = headings[i].split('(')[1].split(')')[0] 
        # print(round((float(depth_id.split('m')[0])-0.145),3))
        new_dict[headings[i]] = {'a1': a1[:, i+2], 'a2': a2[:, i+2], 'a3': a3[:, i+2], 
                              'c1': c1[:, i+2], 'c2': c2[:, i+2], 'c3': c3[:, i+2],
                              'v1': v1[:, i+2], 'v2': v2[:, i+2], 'v3': v3[:, i+2],
                              'avg_speed': csv[headings[i]], 'dtoADCP': round((float(depth_id.split('m')[0])-0.145),3) }
        #print(len(v1[:, i+2]))                     
    new_dict = velocityTotal(new_dict)    
    return new_dict
    
    
def velocityTotal(data_dict):
    headings = data_dict['headings']
    for i in range(0, len(headings)-1):
        velocity_total = []
        for j in range(0, len(data_dict[headings[i]]['v3'])):
            
            velocity_total.append(np.sqrt((data_dict[headings[i]]['v1'][j]**2 + 
                        data_dict[headings[i]]['v2'][j]**2+
                        data_dict[headings[i]]['v3'][j]**2)))
        data_dict[headings[i]]['vtotal'] = velocity_total
    return data_dict
    


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


def SectionData(data_dict, start, end):
    
    DateTime = data_dict['datetime']
    
    # print(DateTime)
    # print(start)
    abs_deltas_from_target_date = np.absolute(DateTime - start)
    index_of_start = np.argmin(abs_deltas_from_target_date)

    abs_deltas_from_target_date = np.absolute(DateTime - end)
    index_of_end = np.argmin(abs_deltas_from_target_date)
    
    return index_of_start, index_of_end

def PlotSection (uptank_dict, downtank_dict, desired_feature, bin_num, start = 0, run_time='nothing'):
    
    for data in [uptank_dict, downtank_dict]:
        heading = data['headings']
        if (start!=0):
            
            end = start+timedelta(minutes=run_time)
            i_start, i_end = SectionData(data, start, end)
            print(data['datetime'][i_end])
            feature = data[heading[bin_num]][desired_feature][i_start:i_end]
            plt.plot(data['datetime'][i_start:i_end],feature/np.linalg.norm(feature), label = data['location'])
            plt.tick_params(axis = 'x', labelrotation=45)
            #plt.locator_params(axis = 'x', tight=True, nbins=30)
        else:
            feature.append(data[heading[bin_num]][desired_feature])
    plt.title(uptank_dict['date'] + heading[bin_num])
    plt.xlabel('time of measurement')
    plt.ylabel('normalized ' + desired_feature)
    plt.legend()

def runComparisonPlots(uptank_dict, downtank_dict, desired_dict, wave_cond, first_bin, title):
    headingup = uptank_dict['headings']
    headingdown = downtank_dict['headings']
    fig, ax = plt.subplots(5, len(wave_cond), figsize =(50,40))
    for i in range (0,5):
        for j in range(0,len(wave_cond)):
            #print(wave_cond[j][2])
            start_i, end_i = SectionData(uptank_dict, wave_cond[j][2], wave_cond[j][3])
            ax[i][j].plot(uptank_dict['datetime'][start_i:end_i], uptank_dict[headingup[first_bin+i]][desired_dict][start_i:end_i], label = uptank_dict['location'])
            start_i, end_i = SectionData(downtank_dict, wave_cond[j][2], wave_cond[j][3])
            ax[i][j].plot(downtank_dict['datetime'][start_i:end_i], downtank_dict[headingdown[first_bin+i]][desired_dict][start_i:end_i], label = downtank_dict['location'])
            ax[i][j].set_title("Ts=" + str(wave_cond[j][0]) + " H=" + str(wave_cond[j][1]))
            ax[i][j].tick_params(axis = 'x', labelrotation=45)
            ax[i][j].set_ylabel(str(uptank_dict[headingup[first_bin+i]]['dtoADCP']) + ' m')
            ax[i][j].legend()
    fig.suptitle(title, y=0.9, fontsize = 20)
    # fig.savefig(title,dpi=300)


    # ax[1].plot(uptank_dict['datetime'], uptank_dict[headingup[3]][desired_dict])
    # ax[1].plot(downtank_dict['datetime'], downtank_dict[headingdown[3]][desired_dict])
    # ax[2].plot(uptank_dict['datetime'], uptank_dict[headingup[4]][desired_dict])
    # ax[2].plot(downtank_dict['datetime'], downtank_dict[headingdown[4]][desired_dict])
    # ax[3].plot(uptank_dict['datetime'], uptank_dict[headingup[5]][desired_dict])
    # ax[3].plot(downtank_dict['datetime'], downtank_dict[headingdown[5]][desired_dict])
    # ax[4].plot(uptank_dict['datetime'], uptank_dict[headingup[6]][desired_dict])
    # ax[4].plot(downtank_dict['datetime'], downtank_dict[headingdown[6]][desired_dict])

    

"""
TODO:
1) Figure out dict, not sure its in the right order
3) Compartison function to find other file at same time and tag them as the same
4) Make pickle of things
* 5) Velocity componets to total velocity
6) Tag section to wave conditions and plot with it
* 7) Figure out bins for thesis and find ones that are the same
* 8) Calibration tests were deleted --> run again? 
9) Comment script
10) Upload to github and clone into sustain repo
""" 





    