import argparse
import torch
import cv2
#import matlab.engine
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
from scipy.ndimage import gaussian_filter
#from skimage.measure import compare_psnr
import os
from math import exp
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description="PyTorch  Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./model_0.0001_epoch_68.pth", type=str, help="model path")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/DRRN_test/mat/X2", type=str, help="image set")
parser.add_argument("--imageset", default="/dataset/1/wangwei/X2/Set5-output-2", type=str, help="image set")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/Urban-output-4", type=str, help="image set")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/Manga109-output-4", type=str, help="image set")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/BSDS100-output-4", type=str, help="image set")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/X2/Set14-output-2", type=str, help="image set")
#parser.add_argument("--imageset", default="/dataset/1/wangwei/test/Set5/Set5-output-4", type=str, help="image set")
parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 2")
'''
转发过来的 PSNR 以及 SSIM
'''
def calc_metrics(img1, img2, crop_border, test_Y=False):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
#    print(psnr)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim

def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_test(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_test(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_test(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        
        

def ssim_test(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


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


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

pathDir =  os.listdir(opt.imageset + "/")
path_psnr = './newpsnr_urban_finall.csv'
path_ssim = './newssim_urban_finall.csv'

path_psnr1 = './psnr_other_urban.csv'
path_ssim1 = './ssim_other_urban.csv'
for i in range(6):
    psnr_test = []
    ssim1_test = []
   
    psnr = []
    ssim1 = []
    psnr_bicubic=0.0
    psnr_predicted=0.0
    
    psnr_bicubic_1=0.0
    psnr_predicted_1=0.0
    psnr_bicubic_average=0.0
    psnr_predicted_average=0.0
    
    psnr_bicubic_average_1=0.0
    psnr_predicted_average_1=0.0
    
    ssim_bicubic = 0.0
    ssim_predicted=0.0
    
    ssim_bicubic_1 = 0.0
    ssim_predicted_1=0.0
    
    ssim_bicubic_average=0.0
    ssim_predicted_average=0.0
    
    ssim_bicubic_average_1=0.0
    ssim_predicted_average_1=0.0
    time_average=0.0
    
    for name in pathDir:
        c=sio.loadmat(opt.imageset + "/" +name)['C']
        if c==3:
            im_gt_y = sio.loadmat(opt.imageset + "/" +name)['im_gt_y']
            im_b_y = sio.loadmat(opt.imageset + "/" +name)['im_b_y']
            im_l_y = sio.loadmat(opt.imageset + "/" +name)['im_l_y']
            im_b_cbcr = sio.loadmat(opt.imageset + "/" +name)['im_b_cbcr']
            #im_b = sio.loadmat(opt.imageset + "/" +name)['im_b']
        else:
            im_gt_y = sio.loadmat(opt.imageset + "/" +name)['im_gt_y']
            im_b_y = sio.loadmat(opt.imageset + "/" +name)['im_b_y']
            im_l_y = sio.loadmat(opt.imageset + "/" +name)['im_l_y']
    
        name = os.path.splitext(name)
        name = name[0]
        #### calculate bicubic psnr & ssim
#        print('im_gt_y',im_gt_y.shape)
#        print('im_gt_y',type(im_gt_y))
        im_gt_y=im_gt_y.astype('float64')
        '''
        保存图像
        '''
        img_file = os.path.join('./img1/','{}.png'.format(name))
        cv2.imwrite(img_file,im_gt_y)
        
        psnr_bicubic_1,ssim_bicubic_1 =calc_metrics(im_b_y,im_gt_y,2)
        psnr_bicubic_average_1 += psnr_bicubic_1
        ssim_bicubic_average_1 += ssim_bicubic_1
        
        psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
        psnr_bicubic_average += psnr_bicubic
        ssim_bicubic= ssim(im_gt_y/255,im_b_y/255)
        ssim_bicubic_average += ssim_bicubic
        
        #### super resolution
        im_input = im_l_y/255.
        im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    #    
    #    if cuda:
    #        model = model.cuda()
    #        a,b,im_input = im_input.cuda()
    #    else:
    #        model = model.cpu()
    #    
    #    start_time = time.time()
    #    im_out= model(im_input[3])
    #    elapsed_time = time.time() - start_time
    #    time_average += elapsed_time
        with torch.no_grad():
            if cuda:
                model = model.cuda()
                im_input= im_input.cuda()
            else:
                model = model.cpu()
            start_time = time.time()
            b,im_out = model(im_input)
        elapsed_time = time.time() - start_time
        time_average += elapsed_time
    
        im_out= im_out[i].cpu()
        out_y = im_out.data[0].numpy().astype(np.float)
        out_y *= 255.0
        out_y = out_y.clip(0., 255.)
#        print('out_y = out_y.clip(0., 255.)',out_y.shape)
        out_y = out_y[0,:,:]
#        print('out_y = out_y[0,:,:]',out_y.shape)
        img_file = os.path.join('./img/','{}.png'.format(name))
        cv2.imwrite(img_file,out_y)
        #### calculate result psnr & ssim
        psnr_predicted_1,ssim_predicted_1 =calc_metrics(out_y,im_gt_y,2)
        psnr_predicted_average_1 += psnr_predicted_1
        ssim_predicted_average_1 += ssim_predicted_1
        
        
        
        psnr_predicted = PSNR(out_y,im_gt_y,shave_border=opt.scale)
        psnr_predicted_average +=psnr_predicted
        ssim_predicted = ssim(im_gt_y/255,out_y/255)
        ssim_predicted_average += ssim_predicted
        #### print the result
        print(name, 'predicted:', psnr_predicted,'bicubic:', psnr_bicubic)
        print(name, 'predicted:', ssim_predicted,'bicubic:', ssim_bicubic)
        
    
    
    
    print('Scale=',opt.scale)
    print('predicted_average=', psnr_predicted_average/len(pathDir), ssim_predicted_average/len(pathDir))
    print('bicubic_average=', psnr_bicubic_average/len(pathDir), ssim_bicubic_average/len(pathDir))
    print("It takes {}s for processing".format(time_average/len(pathDir)))
    
    print('Scale=',opt.scale)
    print('predicted_average=', psnr_predicted_average_1/len(pathDir), ssim_predicted_average_1/len(pathDir))
    print('bicubic_average=', psnr_bicubic_average_1/len(pathDir), ssim_bicubic_average_1/len(pathDir))
    print("It takes {}s for processing".format(time_average/len(pathDir)))
    
    
    psnr.append(psnr_bicubic_average/len(pathDir))
    psnr.append(psnr_predicted_average/len(pathDir))
    ssim1.append(ssim_bicubic_average/len(pathDir))
    ssim1.append(ssim_predicted_average/len(pathDir))

    psnr_test.append(psnr_predicted_average_1/len(pathDir))
    psnr_test.append(psnr_predicted_average_1/len(pathDir))
    ssim1_test.append(ssim_predicted_average_1/len(pathDir))
    ssim1_test.append(ssim_predicted_average_1/len(pathDir))
    

    with open(path_psnr1, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(psnr_test)
    with open(path_ssim1, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ssim1_test)
    with open(path_psnr, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(psnr)
    with open(path_ssim, 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ssim1)

