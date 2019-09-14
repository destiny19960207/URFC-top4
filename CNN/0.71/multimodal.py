import os
import random
import pathlib
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pretrainedmodels

from config import config
from preprocess import *


random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MultiModalDataset(Dataset):
    def __init__(self, images_df, base_path, vis_path, augument=True, mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        if not isinstance(vis_path, pathlib.Path):
            vis_path = pathlib.Path(vis_path)
        self.images_df = images_df.copy()  # csv
        self.augument = augument
        self.vis_path = vis_path  # vist npy path
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / str(x).zfill(6))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):
        if not self.mode == "test":
            y = self.images_df.iloc[index].Target
        else:
            y = str(self.images_df.iloc[index].Id.absolute())


        ### ------- image process --------------
        X = self.read_images(index)
        if self.augument:
            X = process_image_iaa(X)
            X = T.Compose([T.ToPILImage(),  # pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
                           T.ToTensor(),  # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                           ])(X)
        else:
            if self.mode == "tta":
                X = T.Compose([T.ToPILImage(),
                               T.TenCrop(80),  # this is a list of PIL Images
                               T.Lambda(lambda crops: torch.stack([T.Compose([T.Resize(100),
                                                                              T.ColorJitter(),
                                                                              T.ToTensor()])(crop) for crop in crops]))
                               # returns a 4D tensor
                               ])(X)
            else:
                X = T.Compose([T.ToPILImage(),  # pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
                               T.ToTensor(),  # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                               ])(X)

        ### ------- vsiit process --------------
        visit = self.read_npy(index)  # 24x182

        def norm(visit):  # convert ori float npy to 0_1 to then to uint8
            visit = visit.astype(np.float)
            max = np.max(visit)
            min = np.min(visit)
            visit = (visit - min) / (max - min)
            visit = (255 * visit).astype(np.uint8)
            return visit, max, min
        def denorm(visit, max, min):  # convert normed uint8 to ~ori
            visit = visit.astype(np.float) / 255
            visit = visit * (max - min) + min
            return visit
        def std(visit):  # distribution to gauss(0,1)
            visit = visit.astype(np.float)
            return (visit - visit.mean()) / visit.std()
        def Rand_5channels(choice):
            # rand = np.random.choice(6, 3)
            # return [choice[i] for i in list(rand)]
            return [random.choice([choice[0], choice[5]]),
                    random.choice([choice[1], choice[6]]),
                    random.choice([choice[2], choice[7]]),
                    random.choice([choice[3], choice[8]]),
                    random.choice([choice[4], choice[9]])]

        visit = visit.reshape((24, 182, 1))
        visit_l1 = visit / (np.sum(visit, 0, keepdims=True) + 1e-5)
        visit_l2 = visit / (np.sum(visit ** 2, 0, keepdims=True) ** 0.5 + 1e-5)
        visit_log = np.log1p(visit)
        visit, max, min = norm(visit)
        visit_0_1 = visit.astype(np.float) / 255.0
        visit_std = std(visit)

        def gen_new_visit(visit):
            visit_denorm_l1 = denorm(visit, max, min) / (np.sum(denorm(visit, max, min), 0, keepdims=True) + 1e-5)
            visit_denorm_l2 = denorm(visit, max, min) / (np.sum(denorm(visit, max, min) ** 2, 0, keepdims=True) ** 0.5 + 1e-5)
            visit_denorm_log = np.log1p(denorm(visit, max, min))
            visit_denorm_0_1 = norm(denorm(visit, max, min))[0].astype(np.float) / 255.0  # important?
            visit_denorm_std = std(visit)
            choices = [visit_log, visit_0_1, visit_std, visit_l1, visit_l2,
                       visit_denorm_log, visit_denorm_0_1, visit_denorm_std, visit_denorm_l1, visit_denorm_l2]
            cat_list = Rand_5channels(choices)
            visit = np.concatenate(cat_list, 2)
            visit = T.ToTensor()(visit)
            return visit

        if self.augument:
            visit = process_image_visit(visit)
            visit = gen_new_visit(visit)

        else:
            if self.mode == "tta":
                vst_list = []
                visit_befor_aug = visit
                for i in range(10):
                    visit = process_image_visit(visit_befor_aug)
                    visit = gen_new_visit(visit)
                    vst_list.append(visit)
                visit = torch.stack(vst_list)
            else:
                visit = np.concatenate([visit_log, visit_0_1, visit_std, visit_l1, visit_l2], 2)
                visit = T.ToTensor()(visit)

        return X.float(), visit.float(), y

    def read_images(self, index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        images = cv2.imread(filename + '.jpg')
        return images

    def read_npy(self, index):
        row = self.images_df.iloc[index]
        filename = os.path.basename(str(row.Id.absolute()))
        pth = os.path.join(str(self.vis_path.absolute()), filename + '.npy')
        visit = np.load(pth)
        return visit


class FF3DNet(nn.Module):
    def __init__(self,  drop):
        super().__init__()
        img_model = pretrainedmodels.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')  # seresnext101
        self.img_encoder = list(img_model.children())[:-2]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        self.img_encoder = nn.Sequential(*self.img_encoder,
                                         FCViewer(),
                                         nn.Dropout(drop),
                                         nn.Linear(2048, 256),
                                         )

        self.visit_conv = visit_Convnet()

        #### cat 512->9
        cat_dim = 256 + 256
        self.ff_encoder = nn.Sequential(FCViewer(),
                                        nn.ReLU(),
                                        nn.Dropout(drop),
                                        nn.Linear(cat_dim, cat_dim),
                                        nn.ReLU(),
                                        nn.Dropout(drop),
                                        nn.Linear(cat_dim, config.num_classes)
                                        )

    def forward(self, x_img, x_vis):
        x1 = self.img_encoder(x_img)
        x2 = self.visit_conv(x_vis)
        x3 = torch.cat([x1, x2], 1)
        out = self.ff_encoder(x3)
        return out, [x3, None]


class VisitConvNet(nn.Module):
    def __init__(self):
        super(VisitConvNet, self).__init__()
        k = 1
        layer1_1 = []
        layer1_1.append(nn.Conv2d(5, 64 * k, kernel_size=(6, 1), stride=(6, 1)))
        layer1_1.append(nn.BatchNorm2d(64 * k))
        layer1_1.append(nn.ReLU())
        layer1_1.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(1, 7), stride=(1, 7)))
        layer1_1.append(nn.BatchNorm2d(64 * k))
        layer1_1.append(nn.ReLU())
        self.cell_1_1 = nn.Sequential(*layer1_1)
        layer1_2 = []
        layer1_2.append(nn.Conv2d(5, 64 * k, kernel_size=(1, 7), stride=(1, 7), padding=(0, 0)))
        layer1_2.append(nn.BatchNorm2d(64 * k))
        layer1_2.append(nn.ReLU())
        layer1_2.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(6, 1), stride=(6, 1), padding=(0, 0)))
        layer1_2.append(nn.BatchNorm2d(64 * k))
        layer1_2.append(nn.ReLU())
        self.cell_1_2 = nn.Sequential(*layer1_2)
        layer1_3 = []
        layer1_3.append(nn.Conv2d(5, 64 * k, kernel_size=(6, 5), stride=(6, 1), padding=(0, 2)))
        layer1_3.append(nn.BatchNorm2d(64 * k))
        layer1_3.append(nn.ReLU())
        layer1_3.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(5, 7), stride=(1, 7), padding=(2, 0)))
        layer1_3.append(nn.BatchNorm2d(64 * k))
        layer1_3.append(nn.ReLU())
        self.cell_1_3 = nn.Sequential(*layer1_3)
        layer1_4 = []
        layer1_4.append(nn.Conv2d(5, 64 * k, kernel_size=(5, 7), stride=(1, 7), padding=(2, 0)))
        layer1_4.append(nn.BatchNorm2d(64 * k))
        layer1_4.append(nn.ReLU())
        layer1_4.append(nn.Conv2d(64 * k, 64 * k, kernel_size=(6, 5), stride=(6, 1), padding=(0, 2)))
        layer1_4.append(nn.BatchNorm2d(64 * k))
        layer1_4.append(nn.ReLU())
        self.cell_1_4 = nn.Sequential(*layer1_4)


        layer2_1 = []
        layer2_1.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer2_1.append(nn.BatchNorm2d(256 * k))
        layer2_1.append(nn.ReLU())
        layer2_1.append(nn.Dropout(0.1))
        layer2_1.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer2_1.append(nn.BatchNorm2d(256 * k))
        layer2_1.append(nn.ReLU())
        layer2_1.append(nn.Dropout(0.1))
        self.cell_2_1 = nn.Sequential(*layer2_1)
        layer2_2 = []
        layer2_2.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        layer2_2.append(nn.BatchNorm2d(256 * k))
        layer2_2.append(nn.ReLU())
        layer2_2.append(nn.Dropout(0.1))
        layer2_2.append(nn.Conv2d(256 * k, 256 * k, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layer2_2.append(nn.BatchNorm2d(256 * k))
        layer2_2.append(nn.ReLU())
        layer2_2.append(nn.Dropout(0.1))
        self.cell_2_2 = nn.Sequential(*layer2_2)


        layer3_1 = []
        layer3_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer3_1.append(nn.BatchNorm2d(512 * k))
        layer3_1.append(nn.ReLU())
        layer3_1.append(nn.Dropout(0.2))
        layer3_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer3_1.append(nn.BatchNorm2d(512 * k))
        layer3_1.append(nn.ReLU())
        layer3_1.append(nn.Dropout(0.2))
        self.cell_3_1 = nn.Sequential(*layer3_1)


        layer4_1 = []
        layer4_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
        layer4_1.append(nn.BatchNorm2d(512 * k))
        layer4_1.append(nn.ReLU())
        layer4_1.append(nn.Dropout(0.2))
        layer4_1.append(nn.Conv2d(512 * k, 512 * k, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)))
        layer4_1.append(nn.BatchNorm2d(512 * k))
        layer4_1.append(nn.ReLU())
        layer4_1.append(nn.Dropout(0.2))
        self.cell_4_1 = nn.Sequential(*layer4_1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        fc_dim = 4 * 26 * 512 * k
        self.fc = nn.Sequential(FCViewer(),
                                nn.Dropout(0.5),
                                nn.Linear(fc_dim, 512),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(512, 256)
                                )

    def forward(self, x):
        x1_1 = self.cell_1_1(x)
        x1_2 = self.cell_1_2(x)
        x1_3 = self.cell_1_3(x)
        x1_4 = self.cell_1_4(x)
        x_in = torch.cat([x1_1, x1_2, x1_3, x1_4], 1)

        x_out_1 = self.cell_2_1(x_in)
        x_out_2 = self.cell_2_2(x_in)
        x_in = torch.cat([x_out_1, x_out_2], 1)

        x_out = self.cell_3_1(x_in)
        x_in = x_in + x_out

        x_out = self.cell_4_1(x_in)
        x_in = x_in + x_out

        out = self.fc(x_in)
        return out
def visit_Convnet():
    return VisitConvNet()