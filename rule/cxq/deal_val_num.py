import numpy as np
import pandas as pd

val_user_label = pd.read_csv('val_user_label.csv',names=['user_id','label'])
user_label=val_user_label
label_num = user_label.groupby('label')['user_id'].count().reset_index()
label_num.rename(columns = {'user_id':'num'},inplace=True)
S = sum(label_num['num'])
label_num['weight'] = label_num['num'].apply(lambda x:1-x/S)
print (label_num)
id2num = np.load("id2num_val.npy")
w = np.array(label_num['weight'])
for i in range(9):
    id2num[:,i] = id2num[:,i] * w[i]
np.save("new_id2num_val.npy",id2num)