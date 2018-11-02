import os
import pdb
pdb.set_trace()
TXT = 'NPDFace_64add.txt'
TXT1 = 'NPDFace_64add1.txt'
with open(TXT, 'r') as fid:
    lines = fid.readlines()
    with open(TXT1, 'w') as w:
        for line in lines:
            components = line.strip().split(' ')
            imgName = components[0]
            if  os.path.exists(imgName):
                w.write(line)
                #print(imgName)
            #if not os.access(imgName, os.F_OK)