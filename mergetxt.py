#coding=utf-8 
import os
import sys


def subListDir(filepath,steplength):
    path_list=[]
    files = os.listdir(filepath)
    files = sorted(files)
    stepNum = steplength
    for fi in files:
        fi_d = os.path.join(filepath,fi)
        if os.path.isdir(fi_d):
            path_list += subListDir(fi_d,steplength)
        else:
            stepNum-=1
            if stepNum == 0:
                stepNum = steplength
                path_list.append(os.path.join(filepath,fi_d))
    return path_list

def readtxtcontent(path,lennum):
    (filepath,tempfilename) = os.path.split(path)
    fi = str('chg')+tempfilename
    fi_d = os.path.join(filepath,fi)
    f=open(path,'r')
    wf = open(fi_d,'w+')
    for line in f:
        #strf =f.readline()
        listf =[]
        listf= line.split(' ')
        lenf = len(listf)
        if lenf>lennum:
            intlist = []
            for j in listf[1:lenf]:
                intlist.append(int(j))
            realflag = 1
            for i in range(5):  #
                if not(intlist[0]<intlist[4+2*i] and intlist[1]>intlist[4+2*i]):
                    realflag=0
                if not(intlist[2]<intlist[4+2*i+1] and intlist[3]>intlist[4+2*i+1]):
                    realflag=0
            if realflag == 0:
                for i in range(4): #
                    intlist[i]=-1
            imgpath = listf[0]
            b = imgpath.replace('\\','/')           
            wf.writelines(b)
            for n in range(len(intlist)):
                wf.writelines(' ')
                wf.writelines(str(intlist[n]))
            wf.write('\n')
    f.close()
    wf.close()
    return 
    
def alter(file,old_str,new_str):
    file_data = ""
    with open(file, "r") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file,"w") as f:
        f.write(file_data)

if __name__ == '__main__':
    #path = 'V:/NIR_ALLlabeltxt/npdfaceseven/good/allcombine.txt'
    path = 'V:/NIR_ALLlabeltxt/npdfaceseven/npdgood/combine2.txt'
    #path = 'V:/NIR_ALLlabeltxt/npdfaceseven/npdgood/combine1.txt'
    #path = 'V:/NIR_ALLlabeltxt/npdfaceseven/good/combine3.txt'
    #path = 'V:/NIR_ALLlabeltxt/npdfaceseven/good/allcombine.txt'
    readtxtcontent(path,14)
    print('over')
    
