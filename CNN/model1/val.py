import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import math

from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019_id as MM_BDXJTU2019
from Dataloader.MultiModal_BDXJTU2019_for_dense import MM_BDXJTU2019_id as MM_BDXJTU2019_for_dense
from Dataloader.MultiModal_BDXJTU2019_for_nasnet import MM_BDXJTU2019_id as MM_BDXJTU2019_for_nasnet
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50, se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet

import os
import multiprocessing as mp
CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    model_names=["","se_resnext101_32x4d","se_resnext50_32x4d","se_resnet50","densenet169","densenet121"]
    weight_names=["","model1","model2","model3","model_densenet169","model_densenet121"]

    model_id=5
    batch_size=4
    print(model_id)
    model_name=model_names[model_id]
    weight_name=weight_names[model_id]

    weightfiles=os.listdir(weight_name)

    MAX=0
    for file in weightfiles:
        if file.count("_")<3:
            continue
        s=file.split("BDXJTU2019_SGD_")[1]
        s=s.split("_")
        s1=int(s[0])
        s2=int(s[1].split(".")[0])
        if MAX<s1*1000000+s2:
            MAX=s1*1000000+s2
            MAXfile=file
            MAXs1=s1
    print(MAXfile)
    # Dataset
    Dataset = MM_BDXJTU2019("data_txt", mode = 'val')
    if model_name.count("dense")>0:
        Dataset = MM_BDXJTU2019_for_dense("data_txt", mode='val')
    elif model_name.count("nasnet")>0:
        Dataset = MM_BDXJTU2019_for_nasnet("data_txt", mode='val')

    print(batch_size)
    Dataloader = data.DataLoader(Dataset,batch_size,
                                 num_workers=1,
                                 shuffle=False, pin_memory=True)

    # Network
    cudnn.benchmark = True
    # Network = pnasnet5large(6, None)
    # Network = ResNeXt101_64x4d(6)
    net = MultiModalNet(model_name, 'DPN26', 0.5)
    print("weights  model"+str(model_id))
    net.load_state_dict(torch.load(weight_name+'/'+MAXfile))

    net.eval()

    filename = 'val_result/['+str(model_id)+'_'+str(MAXs1)+'].txt'

    import numpy as np

    csvO=open('val_result/csvO['+str(model_id)+'_'+str(MAXs1)+'].csv',"w")

    cnt=0
    name_id=np.zeros([50000])
    for i,(Input_O,visit_tensor, anoss,ids) in enumerate(Dataloader):
        ConfTensor_Os = net.forward(Input_O.cuda(), visit_tensor.cuda())
        for id in range(ConfTensor_Os.shape[0]):
            ConfTensor_O=ConfTensor_Os[id].reshape([1,9])
            anos=[anoss[id]]
            name_id[cnt]=int(ids[id])
            preds_temp = torch.nn.functional.normalize(ConfTensor_O)
            string = ""
            for _ in range(9):
                string = string + str(float(preds_temp[0][_])) + ","
            string = string + "\n"
            csvO.write(string)
            ####################
            preds = torch.nn.functional.normalize(ConfTensor_O)
            _, pred = preds.data.topk(1, 1, True, True)
            if cnt%100==0:
                print(cnt)
            cnt+=1
    print(name_id[:5])
    np.save("name_id"+str(model_id)+".npy",name_id)
    csvO.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    GeResult()

