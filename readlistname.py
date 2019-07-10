import os
import sys


readpath = './person_pngseg1'
writetxt = 'person_pngseg1.txt'

def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    realname = os.path.splitext(name)[0]
                    file.write(realname + "\n")
                    break

def Test():
  dir=readpath     #文件路径
  outfile= writetxt                     #写入的txt文件名
  wildcard = ".jpg .png .jpeg .bmp "      #要读取的文件类型；
  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)

  file.close()


Test()

