# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:34:50 2020

@author: ly
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:14:00 2020

@author: ly
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:04:03 2020

@author: ly
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:24:23 2020

@author: ly
"""


# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np 
import torch.nn.init as init

#from thop import profile

def xavier(param):
    nn.init.xavier_uniform_(param)

  
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                torch.sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
#        print('******////////////',y.shape)
        y = self.conv_du(y)
        y = torch.sigmoid(y)
        return x * y
    
#12.11 将 Resudial Dense 结合在一起 （普通的卷积方式）
class SingleLayer(nn.Module):
    def __init__(self, inChannels,growthRate):
        super(SingleLayer, self).__init__()
        self.conv =nn.Conv2d(inChannels,growthRate,kernel_size=3,padding=1, bias=True)
        self.CA = CALayer(growthRate, growthRate//4)
    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.CA(out))
        out = torch.cat((x, out), 1)
        return out
    
class SingleBlock(nn.Module):
    def __init__(self, inChannels,growthRate,nDenselayer):
        super(SingleBlock, self).__init__()
        self.block= self._make_dense(inChannels,growthRate, nDenselayer)
        
    def _make_dense(self,inChannels,growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels,growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)
                
    def forward(self, x):
        out=self.block(x)
        return out        


'''
'''
   
# 3,24,3,1
# 128,32,3,1
class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel, dilation):
        super().__init__()
        pad_x = int(dilation * (kernel - 1) / 2)
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x, dilation=dilation)

        pad_h = int((kernel - 1) / 2)
        self.conv_hz = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hr = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hn = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

        self.CA = CALayer(oup_dim, oup_dim//4)
#        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, h=None):
        if h is None:
            z = torch.sigmoid(self.conv_xz(x))
            f = torch.tanh(self.conv_xn(x))
            h = z * f
#            print('No')
        else:
            z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
#            print('shifouyunxing  OK')

        h = F.relu(self.CA(h))
        
#        h1 = torch.cat((x,h),1)
        return h, h


class Net(nn.Module):
    def __init__(self,growthRate,nDenselayer):
        super(Net,self).__init__()
        
        # 提取特征图操作  
        self.head = nn.Conv2d(1,64,kernel_size=3, padding=1,bias=True)
        # 2倍下采样操作 得到 64，2h,2w
        self.down2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        # 1 倍 下采样操作 得到 64，h,w
        self.down1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
        inChannels = 64

        # 输出 1倍 图像的 超分
        self.tail0 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3,padding=1, bias=True)
        
        #开始构建 X2倍图像 网络  采用 RDB的连接方式吧 
        # 两个 RDB 块 进行对比 这里面每一层 包含了 Dense CA Concat
        self.DB2_1 = SingleBlock(64,32,8)
        # 每个 RDB 块后面 都加入 1*1 的进行 降维
        self.DB2_1_B = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=1,padding=0, bias=True)
        # 每个 RDB 块 后面加入 ConvGRU CA 模块
        self.GRU2_1 = ConvGRU(64,64,3,1)
        
        # 开始第二个 RDB 模块的构建
        self.DB2_2 = SingleBlock(64,32,8)
        self.DB2_2_B = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=1,padding=0, bias=True)
        self.GRU2_2 = ConvGRU(64,64,3,1)
        
        #开始 第三个 RDB 模块的 构建
        self.DB2_3 = SingleBlock(64,32,8)
        self.DB2_3_B = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=1,padding=0, bias=True)
        self.GRU2_3 = ConvGRU(64,64,3,1)
        
        
        # 将 三个 RDB块的 输出 Concat 一起后 在经过 1*1 卷积 降维 送入 ConvGRU 中 保存状态 
        self.x2_DB_add = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1,padding=0, bias=True)
        self.x2_GRU_add = ConvGRU(64,64,3,1)
        
        # X2 图像 进行亚像素 上采样操作  
        self.HR2 = nn.Sequential(
#                nn.Conv2d(384, out_channels=64, kernel_size=1,padding=0, bias=True),
                nn.Conv2d(64, 4*64, 3, padding=(3-1)//2, stride=1), 
                nn.PixelShuffle(2)                      
                )
        # 输出 2倍 图像的 超分
        self.tail2= nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3,padding=1, bias=True)
        
        # 开始构建 X4 倍 图像 输入：上采样的 x2图像 以及下采样的 x2图像 Concat
        # 两个 RDB 块 进行对比 这里面每一层 包含了 Dense CA Concat
        self.DB4_1 = SingleBlock(128,32,8)
        # 每个 RDB 块后面 都加入 1*1 的进行 降维
        self.DB4_1_B = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1,padding=0, bias=True)
        # 每个 RDB 块 后面加入 ConvGRU CA 模块
        self.GRU4_1 = ConvGRU(128,128,3,1)
        
        # 开始第二个 RDB 模块的构建
        self.DB4_2 = SingleBlock(128,32,8)
        self.DB4_2_B = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1,padding=0, bias=True)
        self.GRU4_2 = ConvGRU(128,128,3,1)
        
        #开始 第三个 RDB 模块的 构建
        self.DB4_3 = SingleBlock(128,32,8)
        self.DB4_3_B = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1,padding=0, bias=True)
        self.GRU4_3 = ConvGRU(128,128,3,1)
        
        
        # 将 三个 RDB块的 输出 Concat 一起后 在经过 1*1 卷积 降维 送入 ConvGRU 中 保存状态 
        self.x4_DB_add = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1,padding=0, bias=True)
        self.x4_GRU_add = ConvGRU(64,64,3,1)
        # x4 的上采样操作 
        self.HR4 = nn.Sequential(
#                nn.ModuleList([ConvGRU(inChannels+i*32,32,3,1) for i in range(8)]),
#                nn.Conv2d(512, out_channels=64, kernel_size=1,padding=0, bias=True),
                nn.Conv2d(64, 4*64, 3, padding=(3-1)//2, stride=1), 
                nn.PixelShuffle(2)                      
                )
        #输出 4倍的图像 
        self.tail4 =  nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3,padding=1, bias=True)
        # 构建下次循环的 输出 
        self.NextInput =  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1,padding=0, bias=True)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

                        
                              
    def forward(self,x):
        
        HR_list = []
        oups = []
        R1 = None
        R2 = None
        R3 = None
        R4 = None
        
        R4_1 = None
        R4_2 = None
        R4_3 = None
        R4_6 = None
        # 2倍 放 大
        Input_x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # 特征 提取层 
        # 通道数 64 
        Input_x =  self.head(Input_x)
        HR0_list = []
        HR2_list = []
        HR4_list = []
        for i in range(6):
            # 保存一下 下次网络的输出 
            next_input = Input_x
            # 下采样 2倍的图像
#            Output_2 = self.down2(Input_x)
            #下采样 1倍的图像 
            Output_1 = self.down1(Input_x)
            
            #开始输出 1 倍 图像
            HR_1 = self.tail0(Output_1)
            HR0_list.append(HR_1)
            
            #开始 构建 X2倍 图像 
            out2_1 = self.DB2_1(Output_1)
            out2_1_B = self.DB2_1_B(out2_1)
            # 外边 局部的 residual link
            out2_1_B = torch.add(out2_1_B,Output_1)
#            print('heuhjbjh',out2_1_B.shape)
            
            out2_1_R,next_out2_1_B = self.GRU2_1(out2_1_B,R1)
            R1 = next_out2_1_B
#            print(out2_1_R.shape)
            #第二个块 
            out2_2 = self.DB2_2(out2_1_R)
            out2_2_B = self.DB2_2_B(out2_2)
            # residual link
            out2_2_B = torch.add(out2_2_B,out2_1_R)
            
            out2_2_R,next_out2_2_B = self.GRU2_2(out2_2_B,R2)
            R2 = next_out2_2_B
            
            #第三个块 
            out2_3 = self.DB2_3(out2_2_R)
            out2_3_B = self.DB2_3_B(out2_3)
            # residual link
            out2_3_B = torch.add(out2_3_B,out2_2_R)
            
            out2_3_R,next_out2_3_B = self.GRU2_3(out2_3_B,R3)
            R3 = next_out2_3_B
            
            # 整合 上述的 RB块 
            x2_finally = torch.cat((out2_1_R,out2_2_R,out2_3_R),1)
            x2_finally = self.x2_DB_add(x2_finally)
            # 用 ConvGRU 保存
            x2_finally,next_x2_finally = self.x2_GRU_add(x2_finally,R4)
            R4 = next_x2_finally
            
            # 上采样 X2 图像 
            HR_2_con =self.HR2(x2_finally) 
            HR_2 = torch.cat((HR_2_con,Input_x),1)
            
            HR_4 = HR_2
            # 作为 第二个阶段的 输入 X4阶段 128 通道 
#            X4_Input = HR_2
            
            HR_2 = self.tail2(HR_2)
            
            HR2_list.append(HR_2)
            
            # 下一次的输入 
            Next = self.NextInput(HR_4)
            
#            HR_4 = self.tail4(HR_4)
#            
#            HR4_list.append(HR_4)
            
            Input_x = Next
            
        
        return HR0_list,HR2_list
            
            
            
            
            
            
            
            

    

model = Net(32,8)
print(model)
x = torch.Tensor(1,1,2,2)

model = Net(32,8)
#print(model)
x = torch.Tensor(1,1,25,25)
#flops, params = profile(model, (x,))
#print('flops: ', flops, 'params: ', params/1000/1000)

HR,a = model(x)
for i in range(len(HR)):
    print(HR[i].shape)
for i in range(len(a)):
    print(a[i].shape)
print(len(HR))
print(len(a))
#print(len(b))
