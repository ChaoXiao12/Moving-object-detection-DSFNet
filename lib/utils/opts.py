from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--model_name', default='DSFNet_with_Static',
                                 help='name of the model. DSFNet_with_Static  |  DSFNet_with_Dynamic  |  DSFNet')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', type=bool, default=False,
                                 help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0,1',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='30,45', #30,45
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=55,  #55
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=5,
                                 help='number of images for per sample. Currently supports 5.')

        # test
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=256,
                                 help='max number of output objects.')
        self.parser.add_argument('--test_large_size', type=bool, default=True,
                                 help='whether or not to test image size of 1024. Only for test.')
        self.parser.add_argument('--show_results', type=bool, default=False,
                                 help='whether or not to show the detection results. Only for test.')
        self.parser.add_argument('--save_track_results', type=bool, default=False,
                                 help='whether or not to save the tracking results of sort. Only for testTrackingSort.')

        # save
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')

        # dataset
        self.parser.add_argument('--datasetname', type=str, default='rsdata',
                                 help='dataset name.')
        self.parser.add_argument('--data_dir', type=str, default= './data/RsCarData/',
                                 help='path of dataset.')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.dataName = opt.data_dir.split('/')[-2]

        now = datetime.now()
        time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

        opt.save_dir = opt.save_dir + '/' + opt.datasetname

        if (not os.path.exists(opt.save_dir)):
            os.mkdir(opt.save_dir)

        opt.save_dir = opt.save_dir + '/' + opt.model_name

        if (not os.path.exists(opt.save_dir)):
            os.mkdir(opt.save_dir)

        opt.save_results_dir = opt.save_dir+'/results'

        opt.save_dir = opt.save_dir + '/weights' + time_str
        opt.save_log_dir = opt.save_dir

        return opt
