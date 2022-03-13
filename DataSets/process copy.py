f = open("DataSets/household_power_consumption.txt")

# change the file to only contain the 4th column
series = []
lines = f.readlines()[1::]
for x in range(len(lines)):
    if lines[x].split(";")[2] == "?":
        lines[x] = lines[x].replace("?", lines[x-1].split(";")[2])

# write series to file
with open("Power_min.csv", "w") as f:
    for item in lines:
        f.write("%s" % item)