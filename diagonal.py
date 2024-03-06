
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import numpy as np
import math






class DAttention(nn.Module):
    
    def __init__(self, in_dim):
        super(DCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        #print(x)
        proj_query = self.query_conv(x)
        m_batchsize, channel, height, width = proj_query.size()
        # print(proj_query.size())
        #proj_query_1 = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize,-1,math.sqrt(width**2+height**2)).permute(0, 2 ,1)
        #proj_query_2 =
        proj_query = proj_query.contiguous().view(m_batchsize*channel,height,width)
        proj_query_diag = torch.diagonal(proj_query,dim1=-2, dim2=-1).contiguous().view(m_batchsize*channel,-1).permute(1,0)
        # proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        #print(proj_query_H)
        # proj_query_sub_d = torch.fliplr(proj_query)
        # proj_query_sub_d = torch.diagonal(proj_query_sub_d, dim1=-2, dim2=-1)
        # proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        #print(proj_query_diag.size())
        proj_key = self.key_conv(x)
        proj_key = proj_key.contiguous().view(m_batchsize * channel, height, width)
        proj_key_diag = torch.diagonal(proj_key, dim1=-2, dim2=-1).contiguous().view(m_batchsize*channel,-1)
        #print(proj_key_diag.size())
        # proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # print(proj_query_H.size())
        # proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        # print(proj_value.size())
        proj_value = proj_value.contiguous().view(m_batchsize * channel, height, width)
        # proj_value_diag = torch.diagonal(proj_value, dim1=-2, dim2=-1)
        # proj_value_diag = proj_value_diag.unsqueeze(0)
        # proj_value_diag = proj_value_diag.repeat(m_batchsize*channel,1,1)
        # print(proj_value_diag.shape)

        # proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        # proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #kk = self.INF(m_batchsize, height, width)
        #print(kk)
        energy_Diag = torch.matmul(proj_query_diag,proj_key_diag)
        energy_Diag = energy_Diag.unsqueeze(0)
        energy_Diag = energy_Diag.repeat(m_batchsize*channel,1,1)
        #print(energy_Diag)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H)+kk).view(m_batchsize,width,height,height).permute(0,2,1,3)
        # # print(energy_H.size())
        # energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        # print(energy_W.size())
        out_D = self.softmax(energy_Diag)
        #print(out_D)
        # print(concate.shape)
        out_D = torch.matmul(proj_value,out_D.permute(0,2,1))
        # print(out_D.shape)
        out_D = out_D.contiguous().view(x.shape)



        # att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        # #print(concate)
        # # print(att_H.size())
        # # print(proj_value_H.size())
        # att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        # #print(att_W)
        # out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        # out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        # print(out_H.size(),out_W.size())
        #print(c)
        return self.gamma*(out_D) + x




