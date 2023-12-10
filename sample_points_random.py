import bz2
import sys
import pandas as pd

in_file = sys.argv[1]
n = int(sys.argv[2])
out_file = sys.argv[3]

df = pd.read_csv(in_file, header=None, index_col=False)
df = df.sample(frac=.05, replace=False)

df.drop(columns=df.columns[-1], inplace=True)
df.reset_index(inplace = True)

indx = df.pop("index")
df = pd.concat([df, indx], axis=1)

print(df.head(3))

df.to_csv(out_file, header=False, index=False)

#with open(in_file, "rt") as fi:
#    with open(out_file, 'w') as fo:
#        count = 0
#        head = True
#        for line in fi:
#            arr = line.rstrip('\n').split(',')
#            fo.write("%s,%s,%d\n" % (arr[1], arr[2], count))
#            count += 1
#            if count == n:
#                break
