f = open("DataSets/power_min.csv")

# change the file to only contain the 4th column
series = []
lines = f.readlines()[1::]
for x in range(0,len(lines), 60):
    hour = [float(p.split(";")[2]) for p in lines[x:x+60]]
    val = sum(hour)/len(hour)
    series.append(val)

# write series to file
with open("Power_hour.csv", "w") as f:
    for item in series:
        f.write("%s\n" % item)