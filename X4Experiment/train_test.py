# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:58:51 2020

@author: ly
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:13:13 2020

@author: ly
"""


# coding: utf-8


import argparse, os
import torch
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from test_model import Net
#from test_for_180 import PSNR,ssim
import time, math
import numpy as np

from dataset import DatasetFromHdf5
import math
import torch.nn.init as init

import tensorflow as tf

import csv
import scipy.io as sio
from scipy.ndimage import gaussian_filter
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#from tensorboardX import SummaryWriter
# Training settings
#batch_size 32代码为 64
parser = argparse.ArgumentParser(description="SR_DenseNet")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--testbatchSize", type=int, default=64, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=600, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0006")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=30")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
opt = parser.parse_args()
print(opt,flush=True)


cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if opt.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed,flush=True)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True      
print("===> Loading datasets",flush=True)
train_set = DatasetFromHdf5("/home/data/wangwei/val_DIV2K_96_4.h5")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#test_set = DatasetFromHdf5("./val_DIV2K_96_4.h5")
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchSize, shuffle=True)
print("===> Building model",flush=True)
#device_ids = [0,1]
model = Net(32,8)
if torch.cuda.is_available():
   model.cuda()
criterion = nn.L1Loss()


print("===> Setting GPU",flush=True)
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
    
# optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))
            
print("===> Setting Optimizer",flush=True)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#optimizer = nn.DataParallel(optimizer,device_ids=device_ids)

sess=tf.Session()
#writer = SummaryWriter()

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2
    
    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    mssim = np.mean(ssim_map)
    
    return mssim

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every step epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(epoch):
    epoch_loss = 0
#    loss = 0
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
#    for param_group in  optimizer.module.param_groups:
#        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]),flush=True)
    
    model.train()    

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        target_0 = input
        target_0 = Variable(target_0, requires_grad=False)
#        target_2 = nn.Upsample(scale_factor=2,
##                                    mode='bicubic', align_corners=False)
        target_2 = F.interpolate(target, scale_factor=1/2, mode='bilinear', align_corners=False)
        target_2 = Variable(target_2, requires_grad=False)
        
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            target_0 = target_0.cuda()
            target_2 = target_2.cuda()
        # 0623 HR 与 LR Add
        # 返回两个值 分别为 HR_list,total_out
        # 返回 HRlist 表，以及各个模块的高频信息  现在高频信息是 1个通道的
        loss = 0
        loss0 =0
        loss2 = 0
        loss4 = 0
        HR0_list,HR2_list,HR4_list = model(input)
        
        for HR_Hope in HR0_list:
            loss0 += criterion(HR_Hope, target_0)
        for HR_Hope in HR2_list:
            loss2 += criterion(HR_Hope, target_2)
        for HR_Hope in HR4_list:
            loss4 += criterion(HR_Hope, target)
        loss = loss0 + loss2 + loss4
        optimizer.zero_grad()
        epoch_loss += float(loss.item())
        loss.backward() 
        optimizer.step()
        if iteration%200== 0:
#            print('loss1',loss1)
#            print('loss2',loss2)
            print("===> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, len(training_data_loader), loss.item()),flush=True)
            

    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, epoch_loss / (len(training_data_loader))),flush=True)


def test(epoch,path):
    # 从模型中获取 参数 
    model = torch.load(path)["model"]
    print('path',path,flush=True)
    
    # 读取 测试文件的 数据集 
    imageset = '/home/data/wangwei/Set5/Set5-output-4'
    
    path_psnr = './psnr.csv'
    path_ssim = './ssim.csv'
    
    
    pathDir =  os.listdir(imageset + "/")
#    print('pathDir',pathDir)
#    print('len(pathDir)',len(pathDir))
    for i in range(6):
        psnr = []
        ssim1 = []
        
        psnr_bicubic=0.0
        psnr_predicted=0.0
        psnr_bicubic_average=0.0
        psnr_predicted_average=0.0

        ssim_bicubic = 0.0
        ssim_predicted=0.0
        ssim_bicubic_average=0.0
        ssim_predicted_average=0.0
        time_average=0.0
        for name in pathDir:
            c=sio.loadmat(imageset + "/" +name)['C']
#            print(c)
            if c==3:
                im_gt_y = sio.loadmat(imageset + "/" +name)['im_gt_y']
                im_b_y = sio.loadmat(imageset + "/" +name)['im_b_y']
                im_l_y = sio.loadmat(imageset + "/" +name)['im_l_y']
                im_b_cbcr = sio.loadmat(imageset + "/" +name)['im_b_cbcr']
            #im_b = sio.loadmat(opt.imageset + "/" +name)['im_b']
            else:
                im_gt_y = sio.loadmat(imageset + "/" +name)['im_gt_y']
                im_b_y = sio.loadmat(imageset + "/" +name)['im_b_y']
                im_l_y = sio.loadmat(imageset + "/" +name)['im_l_y']
    
            name = os.path.splitext(name)
            name = name[0]
            
             #### calculate bicubic psnr & ssim
            im_gt_y=im_gt_y.astype('float64')
            psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=4)
            psnr_bicubic_average += psnr_bicubic
            ssim_bicubic= ssim(im_gt_y/255,im_b_y/255)
            ssim_bicubic_average += ssim_bicubic
            
            #### super resolution
            im_input = im_l_y/255.
            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
            with torch.no_grad():
                if cuda:
                    model = model.cuda()
                    im_input= im_input.cuda()
                else:
                    model = model.cpu()
                start_time = time.time()
                a,b,im_out = model(im_input)
            elapsed_time = time.time() - start_time
            time_average += elapsed_time
            
            im_out= im_out[i].cpu()
            out_y = im_out.data[0].numpy().astype(np.float)
            out_y *= 255.0
            out_y = out_y.clip(0., 255.)
            out_y = out_y[0,:,:]
            #### calculate result psnr & ssim
            psnr_predicted = PSNR(im_gt_y,out_y,shave_border=4)
            psnr_predicted_average +=psnr_predicted
            ssim_predicted = ssim(im_gt_y/255, out_y/255)
            ssim_predicted_average += ssim_predicted
            #### print the result
            print(name, 'predicted:', psnr_predicted,'bicubic:', psnr_bicubic,flush=True)
            print(name, 'predicted:', ssim_predicted,'bicubic:', ssim_bicubic,flush=True)
        print('Epoch=',epoch,flush=True)
        print('What time is ?',i,flush=True)
        print('Epoch:','predicted_average=',psnr_predicted_average/len(pathDir), ssim_predicted_average/len(pathDir),flush=True)
        print('Epoch:bicubic_average=',psnr_bicubic_average/len(pathDir), ssim_bicubic_average/len(pathDir),flush=True)
        print("It takes {}s for processing".format(time_average/len(pathDir)),flush=True)
        
        psnr.append(psnr_bicubic_average/len(pathDir))
        psnr.append(psnr_predicted_average/len(pathDir))
        ssim1.append(ssim_bicubic_average/len(pathDir))
        ssim1.append(ssim_predicted_average/len(pathDir))
        with open(path_psnr, 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(psnr)
        with open(path_ssim, 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ssim1)
            
            
            
def test2(epoch,path):
    # 从模型中获取 参数 
    model = torch.load(path)["model"]
    print('path',path,flush=True)
    
    # 读取 测试文件的 数据集 
    imageset = '/home/data/wangwei/Set5/Set5-output-2'
    
    path_psnr = './psnr2.csv'
    path_ssim = './ssim2.csv'
    
    
    pathDir =  os.listdir(imageset + "/")
#    print('pathDir',pathDir)
#    print('len(pathDir)',len(pathDir))
    for i in range(6):
        psnr = []
        ssim1 = []
        
        psnr_bicubic=0.0
        psnr_predicted=0.0
        psnr_bicubic_average=0.0
        psnr_predicted_average=0.0

        ssim_bicubic = 0.0
        ssim_predicted=0.0
        ssim_bicubic_average=0.0
        ssim_predicted_average=0.0
        time_average=0.0
        for name in pathDir:
            c=sio.loadmat(imageset + "/" +name)['C']
#            print(c)
            if c==3:
                im_gt_y = sio.loadmat(imageset + "/" +name)['im_gt_y']
                im_b_y = sio.loadmat(imageset + "/" +name)['im_b_y']
                im_l_y = sio.loadmat(imageset + "/" +name)['im_l_y']
                im_b_cbcr = sio.loadmat(imageset + "/" +name)['im_b_cbcr']
            #im_b = sio.loadmat(opt.imageset + "/" +name)['im_b']
            else:
                im_gt_y = sio.loadmat(imageset + "/" +name)['im_gt_y']
                im_b_y = sio.loadmat(imageset + "/" +name)['im_b_y']
                im_l_y = sio.loadmat(imageset + "/" +name)['im_l_y']
    
            name = os.path.splitext(name)
            name = name[0]
            
             #### calculate bicubic psnr & ssim
            im_gt_y=im_gt_y.astype('float64')
            psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=4)
            psnr_bicubic_average += psnr_bicubic
            ssim_bicubic= ssim(im_gt_y/255,im_b_y/255)
            ssim_bicubic_average += ssim_bicubic
            
            #### super resolution
            im_input = im_l_y/255.
            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
            with torch.no_grad():
                if cuda:
                    model = model.cuda()
                    im_input= im_input.cuda()
                else:
                    model = model.cpu()
                start_time = time.time()
                a,im_out,b = model(im_input)
            elapsed_time = time.time() - start_time
            time_average += elapsed_time
            
            im_out= im_out[i].cpu()
            out_y = im_out.data[0].numpy().astype(np.float)
            out_y *= 255.0
            out_y = out_y.clip(0., 255.)
            out_y = out_y[0,:,:]
            #### calculate result psnr & ssim
            psnr_predicted = PSNR(im_gt_y,out_y,shave_border=4)
            psnr_predicted_average +=psnr_predicted
            ssim_predicted = ssim(im_gt_y/255, out_y/255)
            ssim_predicted_average += ssim_predicted
            #### print the result
            print(name, 'predicted:', psnr_predicted,'bicubic:', psnr_bicubic,flush=True)
            print(name, 'predicted:', ssim_predicted,'bicubic:', ssim_bicubic,flush=True)
        print('Epoch=',epoch,flush=True)
        print('What time is ?',i,flush=True)
        print('Epoch:','predicted_average=',psnr_predicted_average/len(pathDir), ssim_predicted_average/len(pathDir),flush=True)
        print('Epoch:bicubic_average=',psnr_bicubic_average/len(pathDir), ssim_bicubic_average/len(pathDir),flush=True)
        print("It takes {}s for processing".format(time_average/len(pathDir)),flush=True)
        
        psnr.append(psnr_bicubic_average/len(pathDir))
        psnr.append(psnr_predicted_average/len(pathDir))
        ssim1.append(ssim_bicubic_average/len(pathDir))
        ssim1.append(ssim_predicted_average/len(pathDir))
        with open(path_psnr, 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(psnr)
        with open(path_ssim, 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ssim1)


    
    
def save_checkpoint(epoch):
    model_out_path = "/home/wangwei/Three_RDB/Output/" + "model_{}_epoch_{}.pth".format(opt.lr,epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("/home/wangwei/Three_RDB/Output/"):
        os.makedirs("/home/wangwei/Three_RDB/Output/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path),flush=True)
    return model_out_path

print("===> Training",flush=True)
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    path = save_checkpoint(epoch)
    print('ladys and gentleman conguation!!!!!!!',flush=True)
    test(epoch,path)
    test2(epoch,path)



