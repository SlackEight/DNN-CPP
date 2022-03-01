f = open("DataSets/SP.csv")

# change the file to only contain the 4th column
series = []
for line in f.readlines()[1::]:
    line = line.split(",")
    val = float(line[4])
    series.append(val)

# write series to file
with open("SP_processed.csv", "w") as f:
    for item in series:
        f.write("%s\n" % item)