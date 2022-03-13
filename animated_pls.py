from utils.polar_pla import median_filter, sliding_window_online 

# starting parameters:
time_series_filename = 'DataSets/CTtemp.csv'
max_error = 5900

dataset = []
f = open(time_series_filename, 'r')
for line in f:
    dataset.append(float(line))

dataset = median_filter(dataset, 10)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlim([0, trends[-1][2]])
def animate(i):
    trends = sliding_window_online(dataset[i+1], max_error)
    #i %= (len(trends)-seq_length) # each iteration of the animation will add one more trend
    i+=1
    xs = []
    ys = []
    ax1.clear()
    plt.plot(dataset[0:i], color='orange', alpha=1, linewidth=4)
    for t in trends:
        plt.plot([t[0],t[2]],[t[1],t[3]], color='magenta', linewidth=3)
    
    for t in trends:
        plt.plot([t[0],t[2]],[t[1],t[3]])
    
    #ax1.plot([trends[i][0],trends[i][2]], [trends[i][1],trends[i][3]], color='magenta', linewidth=5, alpha=0.5)
    #ax1.set_xlim([0, trends[-1][2]])
    #plt.xlabel('Date')
    #plt.ylabel('Temp')
    #plt.title('Online PLA')	
	
    
ani = animation.FuncAnimation(fig, animate, interval=0) 
plt.show()