res = input("1:")
ignore = input("2:")
res2 = input("3:")

x = float(res.split(" ")[2])
s = float(res.split(" ")[6])
import math
sa = x
ss = s


x = float(res2.split(" ")[2])
s = float(res2.split(" ")[6])
da = x
ds = s

x1 = float(res.split(" ")[2])
s1 = float(res.split(" ")[6])
aa = (x+x1)/2
aS = (s+s1)/2


latex = f'DNN & {round(da,3)} \pm\ {round(ds,3)} & {round(sa,3)} \pm\ {round(ss,3)} & {round(aa,3)} \pm\ {round(aS,3)} \\\\ \hline'
print(latex)
