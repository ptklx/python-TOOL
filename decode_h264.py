import av
import os
import sys
 
#conda activate pytorch14 
# inputFileName =r"Z:\data\train_egg\egg1m\bug\video.h264"
# inputFileName=r"Z:\data\train_egg\egg1m\train\from\image_new2\3.h264"
# inputFileName='/home/pengtao/data/train_egg/egg1m/train/from/image_new2/5.h264'
# inputFileName='/home/pengtao/data/train_egg/egg1m/train/from/image_new3/5.h264'
# inputFileName='/home/pengtao/data/train_egg/egg1m/train/from/image_new4/2.h264'
# inputFileName=r'Z:\data\train_egg\egg1m\video\hanwei\little_egg\lg321.h264'
# inputFileName='/home/pengtao/data/train_egg/egg1m/video/hanwei/little_egg/lg322.h264'
inputFileName='/home/pengtao/data/train_egg/egg1m/video/hanwei/video1/lg329.h264'
# inputFileName='/home/pengtao/data/train_egg/egg1m/video/overlap/lg321_3.h264'

def my_mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print( "---  new folder...  ---")
    else:
        print( "---  There is this folder!  ---")


def h264ToJpg_demo():
    # inputFileName = "input.h264"

    container = av.open(inputFileName)
    foldpath = os.path.splitext(inputFileName)[0]
    my_mkdir(foldpath)
    print("container:", container)
    print("container.streams:", container.streams)
    print("container.format:", container.format)
 
    for frame in container.decode(video = 0):
        print("process frame: %04d (width: %d, height: %d)" % (frame.index, frame.width, frame.height))
        savename = os.path.join(foldpath,"%d.jpg" % frame.index)
        frame.to_image().save(savename)
 
def main():
    h264ToJpg_demo()
 
 
if __name__ == "__main__":
    sys.exit(main())
