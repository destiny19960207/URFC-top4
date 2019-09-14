import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from Dataloader.MultiModal_BDXJTU2019 import BDXJTU2019_test
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50, se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet

import os

CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']


def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset = BDXJTU2019_test(root='../final')
    batch_size=32
    print(batch_size)
    Dataloader = data.DataLoader(Dataset,batch_size,
                                 num_workers=1,
                                 shuffle=False, pin_memory=True)

    # Network
    cudnn.benchmark = True
    # Network = pnasnet5large(6, None)
    # Network = ResNeXt101_64x4d(6)
    net = MultiModalNet('se_resnext101_32x4d', 'DPN26', 0.5)
    print("weights  model1")
    net.load_state_dict(torch.load('model1/BDXJTU2019_SGD_18_6000.pth'))

    net.eval()

    filename = '[1_18].txt'

    import numpy as np
    f = open(filename, 'w')
    csv6_O=open("csvO[1_18].csv","w")
    csv6_H=open("csvH[1_18].csv","w")
    for (Input_O, Input_H, visit_tensor, anoss) in Dataloader:
        ConfTensor_Os = net.forward(Input_O.cuda(), visit_tensor.cuda())
        ConfTensor_Hs = net.forward(Input_H.cuda(), visit_tensor.cuda())
        # ConfTensor_V = net.forward(Input_V.cuda())
        for id in range(ConfTensor_Os.shape[0]):
            ConfTensor_O=ConfTensor_Os[id].reshape([1,9])
            ConfTensor_H=ConfTensor_Hs[id].reshape([1,9])
            anos=[anoss[id]]
            preds_temp = torch.nn.functional.normalize(ConfTensor_O)
            string = ""
            for _ in range(9):
                string = string + str(float(preds_temp[0][_])) + ","
            string = string + "\n"
            csv6_O.write(string)
            ####################
            ####################
            preds_temp = torch.nn.functional.normalize(ConfTensor_H)
            string = ""
            for _ in range(9):
                string = string + str(float(preds_temp[0][_])) + ","
            string = string + "\n"
            csv6_H.write(string)
            ####################
            preds = torch.nn.functional.normalize(ConfTensor_O) + torch.nn.functional.normalize(
                ConfTensor_H)  # +torch.nn.functional.normalize(ConfTensor_V)
            _, pred = preds.data.topk(1, 1, True, True)

            # f.write(anos[0] + ',' + CLASSES[4] + '\r\n')
            #print(preds[0],"a12343",preds[0].max())


            # cls = pred[0][0]
            # if preds[0].max() < 0.9:
            #     rnd = np.random.randint(1, 100)
            #     if rnd < 50:
            #         cls = 0
            #     elif rnd < 80:
            #         cls = 1
            #     elif rnd < 100:
            #         cls = 5
            #     print(rnd, "True", anos[0][:-4])

            # print(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
            f.writelines(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
            if int(anos[0][:-4])%500==0:
                print(anos[0][:-4])

    csv6_O.close()
    csv6_H.close()
    f.close()


if __name__ == '__main__':
    print("!@#@!$")
    GeResult()

