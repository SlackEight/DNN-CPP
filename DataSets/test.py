
f = open("DataSets/CTtemp.csv")
ts = []
for p in f.readlines():
    p = p.rstrip("\n")
    ts.append(float(p))

def ema(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n/2)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)

    return ema

# return the moving average of a list
def moving_average(ts, window):
    """
    returns a moving average of a time series ts
    using a window of size window

    ts is a list ordered from oldest (index 0) to most
    recent (index -1)
    window is an integer

    returns a numeric array of the moving average
    """
    ts_ma = []
    for i in range(len(ts)):
        start = i - window + 1
        if start < 0:
            start = 0
        ts_ma.append(sum(ts[start:i+1]) / (i+1-start))
    return ts_ma
 

import matplotlib.pyplot as plt
plt.plot(ts)
plt.plot(moving_average(ts,50), linewidth=4)
plt.plot(ema(ts,50),linewidth=4)
plt.show()