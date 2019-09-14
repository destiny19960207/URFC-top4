import pandas as pd
import numpy as np
import os
from time import time
def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):
        return files
    
def getfiles(mode):
    def get_name_from_txt(file_name):
        name_id=[]
        f=open(file_name)
        lines=f.readlines()
        for line in lines:
            s=line.split(".jpg")[0]
            s=s.split("/")[-1]
            name_id.append(s)
        f.close()
        return name_id
    return_ans=[]
    if mode=="test":
        for i in range(2):
            path="/root/userfolder/luotao/final/test_visit_"+str(i)+"/"+str(i)+"/"
            files=os.listdir(path)
            for file in files:
                return_ans.append(path+file)
        return_ans.sort()
        return return_ans
    elif mode=="train":
        file_name="/root/userfolder/luotao/final_code/data_txt/train.txt"
    elif mode=="val":
        file_name="/root/userfolder/luotao/final_code/data_txt/val.txt"
    name_id=get_name_from_txt(file_name)
    name2file={}
    for i in range(10):
        path="/root/userfolder/luotao/final/train_visit_"+str(i)+"/"+str(i)+"/"
        files=os.listdir(path)
        for file in files:
            s=file.split(".txt")[0]
            name2file[s]=path+file
    return_ans=[]
    for name in name_id:
        return_ans.append(name2file[name])
    return return_ans    
def find_key(keys,obj):
    len_keys=len(keys)
    l=0
    r=len_keys-1
    while True:
        if r-l<3:
            break
        mid=(l+r)//2
        if keys[mid]==obj:
            return True
        if keys[mid]<obj:
            l=mid+1
        else:
            r=mid+1
    for k in range(l,r+1):
        if keys[k]==obj:
            return True
    return False    
    
def read_train():
#    读取训练集
    user_label = {'user_id':[],'label':[]}
    tr_files = getfiles("train")
    print(len(tr_files))
    time1=time()
    for i,file in enumerate(tr_files):
       if i % 1000 == 0:
            time2=time()
            print (i,time2-time1)
            time1=time()
       area_label = int(file.split("/")[-1].split('_')[1].split('.txt')[0])
       df = pd.read_csv(file,sep='\t',names=['user_id','info'])
       df['label'] = area_label
       user_label['user_id'].extend(list(df['user_id']))
       user_label['label'].extend(list(df['label']))
    df = pd.DataFrame(user_label,columns=['user_id','label'])
    df.to_csv('train_userid_label.csv',index=False,header=False)
    
def deal_train():
    df = pd.read_csv('train_userid_label.csv',names=['user_id','label'])
    #只保留只去过一个类别的用户
    print (len(df))
    df = df.drop_duplicates()
    print (len(df))
    num_df = df.groupby('user_id').count().reset_index()
    df = df.groupby('user_id')['label'].mean().reset_index()
    tr_df = df[df['user_id'].isin(num_df[num_df['label']==1]['user_id'])].reset_index()
    tr_df = tr_df[['user_id','label']]
    print (tr_df)
    tr_df.to_csv('tr_df.csv',index=False,header=False)
    
def deal_test():
    tr_df = pd.read_csv('tr_df.csv',names=['user_id','label'])
#    读取测试集
    user_label = {'user_id':[],'label':[]}
    ts_files = getfiles("test")
    print (len(ts_files))
    time1 = time()
    for i,file in enumerate(ts_files):
        if i%1000==0:
            time2=time()
            print (i,time2-time1)
            time1=time2
        df = pd.read_csv(file,sep='\t',names=['user_id','info'])
        user_label['user_id'].extend(list(df['user_id']))
    ts_df = pd.DataFrame(user_label,columns=['user_id'])
    
    #规则
    ts_user_label = pd.merge(ts_df,tr_df,on='user_id',how='left')
    ts_user_label = ts_user_label[~ts_user_label['label'].isnull()].groupby('user_id')['label'].mean().reset_index()
    ts_user_label.to_csv('user_label.csv',index=False,header=False)
    
def deal_val():
    tr_df = pd.read_csv('tr_df.csv',names=['user_id','label'])
    #读取验证集
    user_label = {'user_id':[],'label':[]}
    val_files = getfiles("val")
    val_files.sort()
    print (len(val_files))
    time1 = time()
    for i,file in enumerate(val_files):
        if i%1000==0:
            time2=time()
            print (i,time2-time1)
            time1=time2
        df = pd.read_csv(file,sep='\t',names=['user_id','info'])
        user_label['user_id'].extend(list(df['user_id']))
    val_df = pd.DataFrame(user_label,columns=['user_id'])
    
    #规则
    val_user_label = pd.merge(val_df,tr_df,on='user_id',how='left')
    val_user_label = val_user_label[~val_user_label['label'].isnull()].groupby('user_id')['label'].mean().reset_index()
    val_user_label.to_csv('val_user_label.csv',index=False,header=False)

def all_test():
    #测试集submission生成
    ts_user_label = pd.read_csv('user_label.csv',names=['user_id','label'])
    user_label=ts_user_label
    set2=set(user_label['user_id'])
    ul = user_label['user_id']
    test_files = getfiles("test")
    test_files.sort()
    print(len(test_files))
    temps = []
    id2label=np.zeros([100000])
    id2num=np.zeros([100000,9])
    start = time()
    
    for i,file in enumerate(test_files):
        if i % 100 ==0:
            end = time()
            print ('num:'+str(i)+'  time:'+str(end-start))
            start = end
        area_id = int(file.split('.txt')[0].split("/")[-1])
        df = pd.read_csv(file,sep='\t',names=['user_id','info'])
        inter = set(df['user_id']).intersection(set2)
        
        temp=user_label[ul.isin(inter)].groupby('label')['user_id'].count().reset_index()
        temps.append(temp)
        label_cnt=np.zeros([9])
        for j in range(9):
            if len(temp[temp['label']==j+1])==0:
                continue
            label_cnt[j]+=int(temp[temp['label']==j+1]['user_id']) #此处的user_id是类别为j+1的个数
        label=np.argmax(label_cnt)+1
        id2num[area_id,:]=label_cnt[:]
        id2label[area_id]=label
    
    np.save("temps_test.npy",temps)
    np.save("id2label_test.npy",id2label)
    np.save("id2num_test.npy",id2num)

def all_val():
    val_user_label = pd.read_csv('val_user_label.csv',names=['user_id','label'])
    #threshold = 100
    user_label=val_user_label
    cnt=0
    set2=set(user_label['user_id'])
    ul = user_label['user_id']
    val_files = getfiles("val")
    val_files.sort()
    temps = []
    print(len(val_files))
    id2label=np.zeros([50000])
    id2num=np.zeros([50000,9])
    start = time()
    
    for i,file in enumerate(val_files):
        if i % 100 ==0:
            end = time()
            print ('num:'+str(i)+'  time:'+str(end-start),"val acc ",cnt/(i+1))
            start = end
        true_label = int(file.split('.txt')[0].split("/")[-1].split("_")[1])
        df = pd.read_csv(file,sep='\t',names=['user_id','info'])
        inter = set(df['user_id']).intersection(set2)
        
        temp=user_label[ul.isin(inter)].groupby('label')['user_id'].count().reset_index()
        temps.append(temp)
        label_cnt=np.zeros([9])
        for j in range(9):
            if len(temp[temp['label']==j+1])==0:
                continue
            label_cnt[j]+=int(temp[temp['label']==j+1]['user_id']) #此处的user_id是类别为j+1的个数
        label=np.argmax(label_cnt)+1
        id2num[i,:]=label_cnt[:]
        id2label[i]=label
        if true_label==label:
            cnt+=1
            
    np.save("temps_val.npy",temps)
    np.save("id2label_val.npy",id2label)
    np.save("id2num_val.npy",id2num)

if __name__ == '__main__':
    read_train() #读取训练集，获得老用户及其label
    deal_train() #处理老用户，进行一些过滤
    deal_val() #处理验证集，获得验证集中老用户的label
    all_val() #为验证集全部地区根据规则打上label，缺失用1填充
    deal_test() #处理测试集，获得测试集中老用户的label
    all_test() #为测试集全部地区根据规则打上label，缺失用1填充