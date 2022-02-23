from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

  #计算whloss
#ind中记录了目标在heatmap上的地址索引，通过_tranpose_and_gather_feat
# 以及def _gather_feat(feat, ind, mask=None):函数得出我们预测的宽高。
#  _gather_feat根据ind取出feat中对应的元素
def _gather_feat(feat, ind, mask=None):
      #起到的作用是消除各个channel区别的作用，最终得到的inds是对于所有channel而言的。
    #feat（topk_inds）:  batch * (cat x K) * 1      
    #ind（topk_ind）：batch * K
    dim  = feat.size(2)
    #首先将ind扩展一个指标，变为 batch * K * 1
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#    返回的是index：
#feat:  batch * K * 1              取值：[0, cat x K - 1]
#更一般的情况如下：
#feat :  A * B * C      
#ind：A * D
#首先将ind扩展一个指标，并且expand为dim的大小，变为 A * D * C，其中对于任意的i, j, 数组ind[i, j, :]中所有的元素均相同，等于原来A * D shape的ind[i, j]。
#之后使用gather，将ind对应的值取出来。
#得到的feat： A * D * C

    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
   #首先将feat中各channel的元素放到最后一个index中，并且使用contiguous将内存变为连续的，用于后面的view。
#之后将feat变为batch * (W x H) * C的形状
#，使用_gather_feat根据ind取出feat中对应的元素
#返回：
#feat：batch * K * C
#feat[i, j, k]为第i个batch，第k个channel的第j个最大值。

    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count