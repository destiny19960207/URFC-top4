from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.utils import model_zoo

from basenet.senet import senet154,se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d, se_resnext26_32x4d, se_resnet50,se_resnext101_64x4d
#from oct_resnet import oct_resnet26,oct_resnet101
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.nasnet_mobile import nasnetamobile
from basenet.multiscale_resnet import multiscale_resnet
from basenet.multiscale_se_resnext import multiscale_se_resnext
from basenet.multiscale_se_resnext_cat import multiscale_se_resnext_cat
from basenet.DPN import DPN92, DPN26
from basenet.SKNet import SKNet101
from basenet.multiscale_se_resnext_HR import multiscale_se_resnext_HR
from basenet.torchvision_models import densenet169,densenet121,inceptionv3


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MultiModalNet(nn.Module):
    def __init__(self, backbone1, backbone2, drop, pretrained=True):
        super(MultiModalNet, self).__init__()
        self.visit_model = DPN26()

        if backbone1 == 'se_resnext101_32x4d' :
            self.img_encoder = se_resnext101_32x4d(9, None)

            # print("load pretrained model from pth/se_resnext101_32x4d-3b2fe3d8.pth")
            # state_dict = torch.load('pth/se_resnext101_32x4d-3b2fe3d8.pth')
            # state_dict.pop('last_linear.bias')
            # state_dict.pop('last_linear.weight')
            # self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(2048, 256)
        elif backbone1 == 'densenet169':
            self.img_encoder = densenet169(1000, None)
            self.img_fc = nn.Linear(1000, 256)
        elif backbone1 == 'inceptionv3':
            self.img_encoder = inceptionv3(9, None)

            print("load pretrained model from pth inceptionv3")
            state_dict = torch.load('pth/inception_v3_google-1a9a5a14.pth')
            state_dict.pop('fc.bias')
            state_dict.pop('fc.weight')
            self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(1000, 256)
        elif backbone1 == 'densenet121':
                self.img_encoder = densenet121(9, None)

                print("load pretrained model from pth/densenet121-fbdb23505.pth")
                state_dict = torch.load('pth/densenet121-fbdb23505.pth')
                state_dict.pop('classifier.bias')
                state_dict.pop('classifier.weight')
                self.img_encoder.load_state_dict(state_dict, strict=False)

                self.img_fc = nn.Linear(1000, 256)
        elif backbone1 == 'senet154':
            self.img_encoder = senet154(9, None)
            # not right
            # print("load pretrained model from pth/senet154-c7b49a05.pth")
            # state_dict = torch.load('pth/senet154-c7b49a05.pth')
            # state_dict.pop('last_linear.bias')
            # state_dict.pop('last_linear.weight')
            # self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(2048, 256)
        elif backbone1 == 'nasnetalarge':
            self.img_encoder = nasnetalarge(2048,None)

            #not right
            print("load pretrained model from pth/nasnetalarge-a1897284.pth in multimodal.py")
            state_dict = torch.load('pth/nasnetalarge-a1897284.pth')
            #print(state_dict.keys())
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(2048, 256)
        elif backbone1 == 'nasnetamobile':
            self.img_encoder = nasnetamobile(2048, None)
            # not right
            print("load pretrained model from pth nasnetamobile")
            state_dict = torch.load('pth/nasnetamobile-7e03cead.pth')
            # print(state_dict.keys())
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(2048, 256)
        elif backbone1 == 'ResNeXt101_64x4d':
            self.img_encoder = se_resnext101_64x4d(9,None)

            # print("load pretrained model from pth/resnext101_64x4d-e77a0586.pth")
            # state_dict = torch.load('pth/resnext101_64x4d-e77a0586.pth')
            # state_dict.pop('last_linear.bias')
            # state_dict.pop('last_linear.weight')
            # self.img_encoder.load_state_dict(state_dict, strict=False)

            self.img_fc = nn.Linear(2048, 256)
        elif backbone1 == 'se_resnext50_32x4d' :
            self.img_encoder = se_resnext50_32x4d(9, None)
            # print("load pretrained model from pth/se_resnext50_32x4d-a260b3a4.pth")
            # state_dict = torch.load('pth/se_resnext50_32x4d-a260b3a4.pth')

            # print("load pretrained model from weights_82/BDXJTU2019_SGD_82.pth")
            # state_dict1 = torch.load('weights_82/BDXJTU2019_SGD_82.pth')
            #
            # key1=state_dict1.keys()
            # dict_img={}
            # dict_vis={}
            # dict_fc={}
            # dict_cls={}
            #
            # key_img=[]
            # key_vis=[]
            # key_fc=[]
            # key_cls=[]
            # for key in key1:
            #     if key.count("img_encoder")>0:
            #         key_img.append(key)
            #     elif key.count("visit_model")>0:
            #         key_vis.append(key)
            #     elif key.count("img_fc") > 0:
            #         key_fc.append(key)
            #     elif key.count("cls") > 0:
            #         key_cls.append(key)
            #     else:
            #         print(key)
            # dict_img.fromkeys(key_img)
            # dict_vis.fromkeys(key_vis)
            # dict_fc.fromkeys(key_fc)
            # dict_cls.fromkeys(key_cls)
            # for key in key1:
            #     if key.count("img_encoder")>0:
            #         dict_img[key]=state_dict1[key]
            #     elif key.count("visit_model")>0:
            #         dict_vis[key]=state_dict1[key]
            #     elif key.count("img_fc")>0:
            #         dict_fc[key]=state_dict1[key]
            #     elif key.count("cls") > 0:
            #         dict_cls[key]=state_dict1[key]
            #     else:
            #         print(key)

            # state_dict.pop('last_linear.bias')
            # state_dict.pop('last_linear.weight')

            #self.img_encoder.load_state_dict(dict_img, strict = False)
            self.img_fc = nn.Linear(2048, 256)


        elif backbone1 == 'se_resnext26_32x4d' :
            self.img_encoder = se_resnext26_32x4d(9, None)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'multiscale_se_resnext' :
            self.img_encoder = multiscale_se_resnext(9)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'multiscale_se_resnext_cat' :
            self.img_encoder = multiscale_se_resnext(9)
            self.img_fc = nn.Linear(1024, 256)

        elif backbone1 == 'multiscale_se_resnext_HR' :
            self.img_encoder = multiscale_se_resnext_HR(9)
            self.img_fc = nn.Linear(2048, 256)

        elif backbone1 == 'se_resnet50' :
            self.img_encoder = se_resnet50(9, None)
            # print("load pretrained model from pth/se_resnet50-ce0d4300.pth")
            # state_dict = torch.load('pth/se_resnet50-ce0d4300.pth')
            #
            # state_dict.pop('last_linear.bias')
            # state_dict.pop('last_linear.weight')
            # self.img_encoder.load_state_dict(state_dict, strict = False)

            self.img_fc = nn.Linear(2048, 256)

        self.dropout = nn.Dropout(0.5)
        self.cls = nn.Linear(512, 9)

        # self.img_fc.load_state_dict(dict_fc, strict=False)
        # self.visit_model.load_state_dict(dict_vis, strict=False)
        # self.cls.load_state_dict(dict_cls, strict=False)

    def forward(self, x_img,x_vis):
        x_img = self.img_encoder(x_img)
        # print(x_img.shape)
        x_img = self.dropout(x_img)
        x_img = self.img_fc(x_img)
        # print(x_img.shape,"ewe")
        x_vis=self.visit_model(x_vis)

        x_cat = torch.cat((x_img,x_vis),1)
        x_cat = self.cls(x_cat)
        return x_cat
