import bz2
import sys

in_file = sys.argv[1]
n = int(sys.argv[2])
out_file = sys.argv[3]

with open(in_file, "rt") as fi:
    with open(out_file, 'w') as fo:
        count = 0
        head = True
        for line in fi:
            arr = line.rstrip('\n').split(',')
            fo.write("%s,%s,%d\n" % (arr[1], arr[2], count))
            count += 1
            if count == n:
                break
