import cv2
import os

fddb_path = r"D:\data\video"
testset_folder= os.path.join(fddb_path,'video')
testset_list = os.path.join(fddb_path,'box_dets.txt')

with open(testset_list, 'r') as fr:
    # name  = fr.readline()
    test_dataset = fr.read().splitlines()

num_lines = len(test_dataset)
read_n = 0
while read_n <num_lines:
    name = test_dataset[read_n]
    read_n+=1
    if os.path.splitext(name)[1]==".jpg":
        img_path = os.path.join(testset_folder,name)       
        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        box_num = int (float(test_dataset[read_n]))
        for i in range(box_num):
            read_n+=1
            box = test_dataset[read_n]
            box = list(map(float, box.split()))
            b = list(map(int, box[0:4]))
            if box[4]>0.5:  #排除掉不是很相似的
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                text = "{:.4f}".format(box[4])
                cv2.putText(img_raw, text, (b[0], b[1]),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.imshow("detect face",img_raw)
        cv2.waitKey(0)

