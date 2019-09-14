from __future__ import print_function
import os 
import time
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from config import config
from datetime import datetime
from torch import nn,optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score,accuracy_score
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import KFold, StratifiedKFold
from utils import *
from multimodal import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


os.makedirs("./submit_txt/", exist_ok=True)
os.makedirs("./logs/", exist_ok=True)
log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name, mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
log.write('mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    acc  loss  f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------|\n')

def gen_txt(pred_npy, ckpt, fold, best_what):
    submit1 = pd.read_csv(config.test_csv)
    submit1['Predicted'] = np.argmax(pred_npy, 1)
    submit1.drop('Target', axis=1, inplace=True)
    submit1.Predicted = submit1.Predicted.apply(lambda x: "00" + str(int(x) + 1))
    submit1.Id = submit1.Id.apply(lambda x: str(x).zfill(6))
    submit1 = submit1.sort_values('Id', ascending=True)
    submit1.to_csv("./submit_txt/%s_submit_%s_fold%s.txt" % (ckpt["model_name"], best_what, str(fold)), sep='\t', index=None, header=None)
def train(train_loader,model,criterion,optimizer,epoch,valid_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc = AverageMeter()
    model.train()
    for i, (images, visits, target) in enumerate(train_loader):
        visits = visits.to(device)
        images = images.to(device)
        indx_target = target.clone()
        target = torch.from_numpy(np.array(target)).long().to(device)
        #### --------- mixup --------------
        if config.mix_up:
            alpha = config.alpha
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(images.size(0)).cuda()
            images = lam * images + (1 - lam) * images[index, :]
            visits = lam * visits + (1 - lam) * visits[index, :]
            targets_a, targets_b = target, target[index]
            output, features = model(images, visits)
            loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
        else:
            output, features = model(images, visits)
            loss = criterion(output, target)

        psnr = 0
        losses.update(loss.item(),images.size(0))
        f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
        acc_score = accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))
        f1.update(f1_batch,images.size(0))
        acc.update(acc_score,images.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f      |   %0.3f  %0.3f  %0.3f  | %0.3f  %0.3f  %0.4f   | %s  %s  %s |   %s   |   %.2fdB' % (\
                "train", i/len(train_loader) + epoch, epoch,
                acc.avg, losses.avg, f1.avg,
                valid_metrics[0], valid_metrics[1],valid_metrics[2],
                str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                time_to_str((timer() - start),'min'), psnr)
        if i%50==0:
            print(i,message , end='',flush=True)
    log.write("\n")
    #log.write(message)
    #log.write("\n")
    return [acc.avg, losses.avg, f1.avg]
def evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start):
    losses = AverageMeter()
    f1 = AverageMeter()
    acc= AverageMeter()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, visits, target) in enumerate(val_loader):
            images = images.to(device)
            visits = visits.to(device)
            indx_target = target.clone()
            target = torch.from_numpy(np.array(target)).long().to(device)
            if len(images.size())==5:  ### tta
                bs, ncrops, c_i, h_i, w_i = images.size()
                if len(visits.size()) == 4: bs, c_v, h_v, w_v = visits.size()
                if len(visits.size()) == 5: bs, ncrops, c_v, h_v, w_v = visits.size()
                images = images.reshape(-1, c_i, h_i, w_i)
                visits = visits.reshape(-1, c_v, h_v, w_v).contiguous()
                output, features = model(images, visits)
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output, features = model(images,visits)
            loss = criterion(output,target)
            psnr = 0
            losses.update(loss.item(),images.size(0))
            f1_batch = f1_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1),average='macro')
            acc_score=accuracy_score(target.cpu().data.numpy(),np.argmax(F.softmax(output).cpu().data.numpy(),axis=1))        
            f1.update(f1_batch,images.size(0))
            acc.update(acc_score,images.size(0))
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f     |     %0.3f  %0.3f   %0.3f    | %0.3f  %0.3f  %0.4f  | %s  %s  %s  |  %s   |   %.2fdB' % (\
                    "val", i/len(val_loader) + epoch, epoch,                    
                    acc.avg,losses.avg,f1.avg,
                    train_metrics[0], train_metrics[1],train_metrics[2],
                    str(best_results[0])[:8],str(best_results[1])[:8],str(best_results[2])[:8],
                    time_to_str((timer() - start),'min'),psnr)
            print(message, end='',flush=True)
        log.write("\n")
        #log.write(message)
        #log.write("\n")
    return [acc.avg, losses.avg, f1.avg]
def test(test_loader, model, fold, ckpt, best_what, if_gen_txt):
    save_dir = os.path.join('./preds_9', config.model_name)
    os.makedirs(save_dir, exist_ok=True)
    predicts = []
    model.to(device)
    model.eval()
    for i, (image, visit, _) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            image = image.to(device)
            visit = visit.to(device)
            if len(image.size()) == 5:
                bs, ncrops, c_i, h_i, w_i = image.size()
                if len(visit.size()) == 4: bs, c_v, h_v, w_v = visit.size()
                if len(visit.size()) == 5: bs, ncrops, c_v, h_v, w_v = visit.size()

                image = image.reshape(-1, c_i, h_i, w_i)
                visit = visit.reshape(-1, c_v, h_v, w_v).contiguous()
                output, features = model(image,visit)
                y_pred = output.view(bs, ncrops, -1).mean(1)
            else: y_pred, _ = model(image,visit)
            y_pred=F.softmax(y_pred).cpu().data.numpy()
            predicts.append(y_pred)
    pred_npy = np.concatenate(predicts)
    save_name = '%s_val_%.4f_fold%d_%s.npy'%(ckpt["model_name"], ckpt["best_acc"], fold, best_what)
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, pred_npy)
    if if_gen_txt:
        gen_txt(pred_npy, ckpt, fold, best_what)
def test_ensemble_loss_acc(test_loader, fold, ckpt, best_what, if_gen_txt):
    save_dir = os.path.join('./preds_9', config.model_name)
    os.makedirs(save_dir, exist_ok=True)
    loss_pred = np.load('%s/%s_val_%.4f_fold%d_%s.npy'
                        %(save_dir, ckpt[0]["model_name"], ckpt[0]["best_acc"], fold, 'best_loss'))
    acc_pred = np.load('%s/%s_val_%.4f_fold%d_%s.npy'
                       %(save_dir, ckpt[1]["model_name"], ckpt[1]["best_acc"], fold, 'best_acc'))
    pred_npy = (loss_pred + acc_pred) / 2

    save_name = '%s_val_fold%d_%s.npy'%(ckpt[0]["model_name"], fold, best_what)
    save_path = os.path.join(save_dir, save_name)
    np.save(save_path, pred_npy)
    if if_gen_txt:
        gen_txt(pred_npy, ckpt[0], fold, best_what)


def training(train_data_list, val_data_list, test_files, fold):

    os.makedirs(os.path.join(config.weights, config.model_name) + os.sep + str(fold), exist_ok=True)
    os.makedirs(config.best_models, exist_ok=True)
    ### ---------- get model ------------------------------------------
    model = FF3DNet(drop=0.5)
    ### ---------- set lr, opt, loss ------------------------------------------
    img_params = list(map(id, model.img_encoder.parameters()))
    rest_params = filter(lambda p: id(p) not in img_params, model.parameters())
    params = [{'params': rest_params, 'lr': config.lr},
              {'params': model.img_encoder.parameters(), 'lr': config.lr * 3},
              ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    #optimizer=torch.optim.Adam(params,lr=0.0001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - 5, eta_min=config.lr / 100)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=5, after_scheduler=scheduler)

    criterion = nn.CrossEntropyLoss().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    best_results = [0, np.inf, 0]
    val_metrics = [0, np.inf, 0]
    ### ---------- load dataset ------------------------------------------
    train_gen = MultiModalDataset(train_data_list, config.train_data, config.train_vis, mode="train")
    train_loader = DataLoader(train_gen, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_gen = MultiModalDataset(val_data_list, config.train_data, config.train_vis, augument=False, mode="val")
    val_loader = DataLoader(val_gen, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_gen = MultiModalDataset(test_files, config.test_data, config.test_vis, augument=False, mode="test")
    test_loader = DataLoader(test_gen, 20, shuffle=False, pin_memory=True, num_workers=4)

    # --- train, val, test -------------------------
    resume =False

    start = timer()
    #___________________________________________________________________________________________________________________
    checkpoint_loss = torch.load('checkpoints/best_models/0626_debug_fold_111_model_best_loss.pth.tar')
    model.load_state_dict(checkpoint_loss["state_dict"])
    # #test(test_loader, model, fold, checkpoint_loss, 'best_loss', False)

    checkpoint_acc = torch.load('checkpoints/best_models/0626_debug_fold_111_model_best_acc.pth.tar')
    model.load_state_dict(checkpoint_acc["state_dict"])
    # #test(test_loader, model, fold, checkpoint_acc, 'best_acc', False)
    # #test_ensemble_loss_acc(test_loader, fold, [checkpoint_loss, checkpoint_acc], 'ensemble', True)
    #___________________________________________________________________________________________________________________

    if resume:
        checkpoint_loss = torch.load('checkpoints/best_models/0616_coslr_55_fold_0_model_best_loss.pth.tar')
        model.load_state_dict(checkpoint_loss["state_dict"])
        test(test_loader, model, fold, checkpoint_loss, 'best_loss', False)
        checkpoint_acc = torch.load('checkpoints/best_models/0616_coslr_55_fold_0_model_best_acc.pth.tar')
        model.load_state_dict(checkpoint_acc["state_dict"])
        test(test_loader, model, fold, checkpoint_acc, 'best_acc', False)
        test_ensemble_loss_acc(test_loader, fold, [checkpoint_loss, checkpoint_acc], 'ensemble', True)
    else:
        ### ---------- train loop ----------------
        if fold==1:
            print("fold==1 0728")
            start_epoch=0
        else:
            start_epoch=0
        for epoch in range(start_epoch, config.epochs):
            scheduler_warmup.step(metrics=val_metrics[0])
            for param_group in optimizer.param_groups:
                log.write(str(param_group['lr'])+'\n')
            train_metrics = train(train_loader, model, criterion, optimizer, epoch, val_metrics, best_results, start)
            # val_metrics_tta = evaluate(val_loader_tta,model,criterion,epoch,train_metrics,best_results,start)
            val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, best_results, start)
            is_best_acc = val_metrics[0] > best_results[0]
            best_results[0] = max(val_metrics[0], best_results[0])
            is_best_loss = val_metrics[1] < best_results[1]
            best_results[1] = min(val_metrics[1], best_results[1])
            is_best_f1 = val_metrics[2] > best_results[2]
            best_results[2] = max(val_metrics[2], best_results[2])
            save_checkpoint({
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "best_acc": best_results[0],
                "best_loss": best_results[1],
                "optimizer": optimizer.state_dict(),
                "fold": fold,
                "best_f1": best_results[2],
            }, is_best_acc, is_best_loss, is_best_f1, fold)
            print('\r', end='', flush=True)
            print(val_metrics[0], val_metrics[1], val_metrics[2],"val")
            log.write(
                '%s  %5.1f %6.1f      |   %0.3f   %0.3f   %0.3f     |  %0.3f   %0.3f    %0.3f    |   %s  %s  %s | %s' % ( \
                    "best", epoch, epoch,
                    train_metrics[0], train_metrics[1], train_metrics[2],
                    val_metrics[0], val_metrics[1], val_metrics[2],
                    str(best_results[0])[:8], str(best_results[1])[:8], str(best_results[2])[:8],
                    time_to_str((timer() - start), 'min'))
                )
            log.write("\n")
            time.sleep(0.001)
        # log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        # datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
        # log.write(
        #     '                           |------------ Train -------|----------- Valid ---------|----------Best Results---|------------|\n')
        # log.write(
        #     'mode     iter     epoch    |    acc  loss  f1_macro   |    acc  loss  f1_macro    |    acc  loss  f1_macro       | time       |\n')
        # log.write(
        #     '-------------------------------------------------------------------------------------------------------------------------|\n')

        ### ---------- per fold ensemble best loss ckpt and best acc ckpt
        checkpoint_loss = torch.load('checkpoints/best_models/%s_fold_%s_model_best_loss.pth.tar'% (config.model_name, str(fold)))
        model.load_state_dict(checkpoint_loss["state_dict"])
        test(test_loader, model, fold, checkpoint_loss, 'best_loss', False)
        checkpoint_acc = torch.load('checkpoints/best_models/%s_fold_%s_model_best_acc.pth.tar'% (config.model_name, str(fold)))
        model.load_state_dict(checkpoint_acc["state_dict"])
        test(test_loader, model, fold, checkpoint_acc, 'best_acc', False)
        test_ensemble_loss_acc(test_loader, fold, [checkpoint_loss, checkpoint_acc], 'ensemble', not config.k_fold)

    ### ----------- last kfold ensemble all before k ensemble ckpts
    if config.k_fold and fold == config.num_kf:
        mean_npy = np.zeros([10000, 9])
        for i in range(1, config.num_kf+1):
            checkpoint = torch.load('checkpoints/best_models/%s_fold_%s_model_best_loss.pth.tar'% (config.model_name, str(i)))
            loss_pred = np.load('preds_9/%s/%s_val_fold%s_%s.npy'% (checkpoint["model_name"], checkpoint["model_name"],  str(i), 'ensemble'))
            mean_npy += loss_pred
        mean_npy = mean_npy/config.num_kf
        np.save('preds_9/%s/%s_val_fold%s_%s.npy'% (checkpoint["model_name"], checkpoint["model_name"], 'cv', 'ensemble'), mean_npy)
        gen_txt(mean_npy, checkpoint, 'cv', 'ensemble')


def main():
    fold = 0
    all_files = pd.read_csv(config.train_csv)
    test_files = pd.read_csv(config.test_csv)
    
    ### -------- kfold or not kfold ---------------
    if not config.k_fold:
        train_data_list, val_data_list = train_test_split(all_files, test_size=0.1, random_state = 2050)

        training(train_data_list, val_data_list, test_files, fold)
    else:
        print(" K fold ",config.num_kf)
        kf = StratifiedKFold(n_splits=config.num_kf, shuffle=True)
        for train_index, test_index in kf.split(all_files, all_files.loc[:,"Target"]):
            fold +=1
            if fold>1:
                continue
            message_start_kf = "*" * 50 + " KFold_%d " % fold + "*" * 50+'\n'
            log.write(message_start_kf)
            train_data_list = all_files.loc[train_index]
            val_data_list = all_files.loc[test_index]
            training(train_data_list, val_data_list, test_files, fold)


if __name__ == "__main__":
    main()
