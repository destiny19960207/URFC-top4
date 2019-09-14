import os

path_img="../../final/train_image/00"

f=open("all.txt","w")
for i in range(1,10):
    path=path_img+str(i)+"/"
    files=os.listdir(path)
    for file in files:
        f.write(path[3:]+file+"\n")
f.close()