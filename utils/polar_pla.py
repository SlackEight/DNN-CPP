from time import time
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.lib.function_base import average
from matplotlib.lines import Line2D
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
import random
#import utils.segment as segment
#import utils.fit as fit

def simple_moving_average(time_series, window_size):
    moving_average = []
    for i in range(len(time_series)):
        if (i<window_size):
            moving_average.append(sum(time_series[:i+1])/(i+1))
        else: 
            moving_average.append(sum(time_series[i+1-window_size:i+1])/window_size)
    
    return moving_average

values = [0,0,0,0,0,0,0,0] # E_x, E_y, E_xy, E_x2, E_y2, m, b, loss
trends = []
current_trend_index = -1
def sliding_window_online(point, max_error):
    # A trend looks like this: [start x, start y, end x, end y]
    global trends
    global values
    global current_trend_index
    
    if len(trends) == 0:  # after first run we have one point, add it
        trends.append([0, point, 0, point])
        current_trend_index = 0
        #values = [0.5, point[1], point[0]*point[1], point[0]**2, point[1]**2, 0, 0, 0]

    elif trends[-1][0] == trends[-1][2]: # in the case our current latest trend has only one point, make it a line with the next point
        point = [trends[-1][2]+1, point]
        trends[-1] = [trends[-1][0],trends[-1][1], point[0], point[1]]
        values[0] = values[0] + 0.5 # E_x
        values[1] = (trends[-1][1] + point[1])/2 # E_y
        values[2] = (trends[-1][1]*trends[-1][0] + point[1]*point[0])/2 # E_xy
        values[3] = (trends[-1][0]**2+point[0]**2)/2 # E_x2
        values[4] = (trends[-1][1]**2+point[1]**2)/2 # E_y2
        values[5] = (point[1]-trends[-1][1]) # m
        values[6] = (point[1]-values[5]*point[0]) # b
        values[7] = 0 # loss
            #print trends with each value rounded to 2 decimal places


    else: # otherwise we're in the middle of a normal trend
        current_trend = trends[-1]
        window_size = current_trend[2] - current_trend[0] + 2
        N = window_size-1
        new_x = current_trend[2]+1
        new_y = point
        E_x = values[0]*N/(N+1) + new_x*1/(N+1)
        E_y = values[1]*N/(N+1) + new_y*1/(N+1)
        E_xy = values[2]*N/(N+1) + new_x*new_y/(N+1)
        E_x2 = values[3]*N/(N+1) + new_x*new_x/(N+1)
        E_y2 = values[4]*N/(N+1) + new_y*new_y/(N+1)
        t_m = (E_x*E_y-E_xy)/(E_x**2-E_x2)
        t_b = E_y-t_m*E_x
        loss = E_y2 - 2 * (t_m * E_xy + t_b * E_y) + t_m**2 * E_x2 + 2 * t_m * t_b * E_x + t_b**2
        if loss*window_size < max_error:
            m = t_m
            b = t_b
            values = [E_x, E_y, E_xy, E_x2, E_y2, m, b, loss]
            trends[-1] = [current_trend[0], b+m*current_trend[0], current_trend[0]+window_size-1, b+m*(current_trend[2]+1)]

        else:
            # trend is over finalise previous trend and add the new point to a new trend
            b = values[6]
            m = values[5]
            trends.append([trends[-1][2],trends[-1][3], new_x, new_y])

            point = [trends[-1][2], point]
            values[0] =  (trends[-1][0]+trends[-1][2])/2 # E_x
            values[1] = (trends[-1][1] + point[1])/2 # E_y
            values[2] = (trends[-1][1]*trends[-1][0] + point[1]*point[0])/2 # E_xy
            values[3] = (trends[-1][0]**2+point[0]**2)/2 # E_x2
            values[4] = (trends[-1][1]**2+point[1]**2)/2 # E_y2
            values[5] = (point[1]-trends[-1][1]) # m
            values[6] = (point[1]-values[5]*point[0]) # b
            values[7] = 0 # loss
            current_trend_index += 1
            #print(current_trend_index)

        #trends.append([anchor, b, anchor+window_size, (time_series[anchor]+b+(window_size)*m)])
    output = trends
    return output

def convert_trend_representation(trends):
    '''
        Takes a list of trends where each trend is in the form [startX, startY, endX, endY]
        Returns a list where each trend is represented as [slope, duration]
    '''
    result = []
    for trend in trends:
        # find the slope in degrees
        slope = math.degrees(math.atan((trend[3]-trend[1])/(trend[2]-trend[0])))
        #slope = (trend[3]-trend[1])/(trend[2]-trend[0])
        
        # now we need to normalize the angle to a value from -1 to 1
        #slope /= 90
        
        duration = trend[2]-trend[0]
        
        #result.append([slope,duration])
        result.append([slope,duration])
    return result

def preprocess(file_name, filter_size, pls_max_error, seq_length, component):
    '''
        Performs all data preprocessing steps and returns a processed trend series
    '''
    global trends
    global current_trend_index
    global values
    
    # read in the time series
    f = open(file_name, 'r')
    time_series = []
    for line in f:
        time_series.append(float(line))
    #time_series = time_series[:10000]
    if filter_size > 0:
        #time_series = median_filter(time_series, filter_size)
        time_series = simple_moving_average(time_series, filter_size)
    
    time_series_normalized = []
    # do minmax normalization on the time series 
    for i in range(len(time_series)):
        time_series_normalized.append((time_series[i] - min(time_series))/(max(time_series)-min(time_series)))


    finished_pla = []
    
    for point in time_series:
        sliding_window_online(point, pls_max_error)
    
    finished_pla = trends
    print("TOTAL TRENDS: " + str(len(finished_pla)))
    values = [0,0,0,0,0,0,0,0] # E_x, E_y, E_xy, E_x2, E_y2, m, b, loss
    trends = []
    current_trend_index = -1

    inputs = []
    outputs = []
    index = 0
    
    yuh = []
    for x in range(seq_length):
        sliding_window_online(time_series[index], pls_max_error)
        index += 1
    
    while(len(trends) < seq_length):
        yuh = sliding_window_online(time_series[index], pls_max_error)
        index += 1

    #for x in range(len(time_series)-index-5000):
    max_len = 0
    for t in finished_pla:
        if (t[2]-t[0]) > max_len:
            max_len = t[2]-t[0]
    print("Longest trend line length: " + str(max_len))

    yuh = sliding_window_online(time_series[index], pls_max_error)
    index += 1
    # forming of input output pairs
    while current_trend_index < len(finished_pla)-1:

        temp = convert_trend_representation(trends[current_trend_index-seq_length+1:current_trend_index+1])
        
        #for p in time_series_normalized[index-7:index+1]:
        #    temp.append([p, 0])
        
        #[[s1, d1], [s2, d2], [s3, d3], [s4, d4], [s5, d5], [s6, d6], [s7, d7], [s8, d8]]
        
        inputs.append(temp)
        #temp = trends[current_trend_index-seq_length+1:current_trend_index]
        #temp.append(finished)
        #inputs.append(convert_trend_representation(, max_len))

        current_trend = finished_pla[current_trend_index]
        next_trend = finished_pla[current_trend_index+1]

        current_trend_duration = current_trend[2]-current_trend[0]
        online_trend_duration = trends[-1][2]-trends[-1][0]
        
        remaining_duration = current_trend_duration - online_trend_duration


        startX, startY, endX, endY = next_trend[0], next_trend[1], next_trend[2], next_trend[3]
        next_trend_angle = math.degrees(math.atan((endY-startY)/(endX-startX)))#/90

        #outputs.append([next_trend_angle/90, remaining_duration/max_len])
        if component == 0:
            #outputs.append([next_trend_angle])
            outputs.append([next_trend_angle])
        elif component == 1:
            outputs.append([remaining_duration])
        elif component == 2:
            outputs.append([next_trend_angle, remaining_duration])
            
        yuh = sliding_window_online(time_series[index], pls_max_error)
        index += 1
    
    #print("FINAL TRENDS", convert_trend_representation(trends, 1))
    #print()
    #print("INPUTS ", inputs)
    #print()
    #print("OUTPUTS ", outputs)
    plt.plot(time_series, color='orange', alpha=1, linewidth=4)
    for t in finished_pla:
        plt.plot([t[0],t[2]],[t[1],t[3]], color='magenta', linewidth=3)
    
    for t in yuh:
        plt.plot([t[0],t[2]],[t[1],t[3]])
    #plot the time series in purple
    #plt.ion()
    plt.show()
    
    return inputs, outputs

def preprocess_from_pickle(file_name, seq_length, component):
    '''
        Performs all data preprocessing steps and returns a processed trend series
    '''
    import pickle

    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    trends = data.iloc[:,0:2].values
    inputs = []
    outputs = []
    for i in range(seq_length, len(trends)-1):
        inputs.append(trends[i-seq_length+1:i+1].tolist())
        
        if component == 0:
            outputs.append([trends[i+1].tolist()[0]])
        elif component == 1:
            outputs.append([trends[i+1].tolist()[1]])
        elif component == 2:
            outputs.append(trends[i+1].tolist())
    return inputs, outputs
    
    
    
    
    return 1#inputs, outputs

if __name__ == "__main__":
    f = open('DataSets/CTtemp.csv')
    data = f.readlines()[0:1000]
    f.close()
    ts = []
    for line in data:
        ts.append(float(line))
    #plt.plot(ts)
    #ts = median_filter(ts, 5)
    sample = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,15,14,13,12,11,10,9,8,9,10,11,12,13,12,11,10,9,8,7,6,5,4,3,2,1,3,5,7,9,11,13]
    ts = sample
    #trends = sliding_window(ts, 6000)#PiecewiseLinearSegmentation.Sliding(0).transform(ts)#
    #print(trends)
    plt.plot(ts, color='blue',linewidth=3)
    #for t in trends:
    #    plt.plot([t[0],t[2]],[t[1],t[3]], color='orange')
    #plt.show()
    for i in range(18):
        #print(f"\n\n                                                                 Iteration {i+1}:")
        test = sliding_window_online(ts[i], 0)
        #print("trends:",trends)
        #print("values:",values)
        #print(f"\n\n")
    plt.plot(ts, color='blue',linewidth=1)
    for t in test:
        plt.plot([t[0],t[2]],[t[1],t[3]], color='orange')
    for trend in test:
        print([ '%.2f' % elem for elem in trend ])
    plt.show()
    


