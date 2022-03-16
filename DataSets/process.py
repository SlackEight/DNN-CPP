f = open("DataSets/N225.csv")

# change the file to only contain the 4th column
series = []
lines = f.readlines()[1::]
for x in range(len(lines)):
    v = lines[x].split(",")[4]
    if v == "null":
        series.append(series[-1])
    else:
        series.append(float(v))

# write series to file
with open("N225P.csv", "w") as f:
    for item in series:
        f.write("%s\n" % item)