from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch

from lib.utils.opts import opts

from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.coco import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

import cv2

from progress.bar import Bar

import time

import scipy.io as scio

CONFIDENCE_thres = 0.3
COLORS = [(255, 0, 0)]

FONT = cv2.FONT_HERSHEY_SIMPLEX

def cv2_demo(frame, detections):
    det = []
    for i in range(detections.shape[0]):
        if detections[i, 4] >= CONFIDENCE_thres:
            pt = detections[i, :]
            cv2.rectangle(frame,(int(pt[0])-4, int(pt[1])-4),(int(pt[2])+4, int(pt[3])+4),COLORS[0], 2)
            cv2.putText(frame, str(pt[4]), (int(pt[0]), int(pt[1])), FONT, 1, (0, 255, 0), 1)
            det.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3]),detections[i, 4]])
    return frame, det

def process(model, image, return_time):
    with torch.no_grad():
        output = model(image)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wh, reg=reg)
    if return_time:
        return output, dets, forward_time
    else:
        return output, dets

def post_process(dets, meta, num_classes=1, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def pre_process(image, scale=1):
    height, width = image.shape[2:4]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = height, width
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    meta = {'c': c, 's': s,
            'out_height': inp_height ,
            'out_width': inp_width}
    return meta

def merge_outputs(detections, num_classes ,max_per_image):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

        soft_nms(results[j], Nt=0.5, method=2)

    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results

def test(opt, split, modelPath, show_flag, results_name, saveMat=False):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # Logger(opt)
    print(opt.model_name)

    dataset = COCO(opt, split)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name, opt)  # 建立模型
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    results = {}

    return_time = False
    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt.K

    if saveMat:
        save_mat_path_upper = os.path.join(opt.save_results_dir, results_name+'_mat')
        if not os.path.exists(save_mat_path_upper):
            os.mkdir(save_mat_path_upper)

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # print(ind)
        if(ind>num_iters):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
        )

        start_time = time.time()

        #read image
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)
        image = pre_processed_images['input'].cuda()
        img = pre_processed_images['imgOri'].squeeze().numpy()
        if saveMat:
            file_name = pre_processed_images['file_name']
            mat_name = file_name[0].split('/')[-1].replace('.jpg', '.mat')
            save_mat_folder = os.path.join(save_mat_path_upper, file_name[0].split('/')[2])
            if not os.path.exists(save_mat_folder):
                os.mkdir(save_mat_folder)

        #det
        output, dets = process(model, image, return_time)

        #post process
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        end_time = time.time()
        # print('process time:', end_time-start_time)

        if(show_flag):
            frame, det = cv2_demo(img, dets[1])

            cv2.imshow('frame',frame)
            cv2.waitKey(5)

            hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()

            cv2.imshow('hm', hm1)
            cv2.waitKey(5)

        if (saveMat):
            matsaveName = os.path.join(save_mat_folder, mat_name)
            A = np.array(ret[1])
            scio.savemat(matsaveName, {'A': A})

        results[img_id.numpy().astype(np.int32)[0]] = ret
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_results_dir, results_name)

if __name__ == '__main__':
    opt = opts().parse()

    split = 'test'

    show_flag = False

    save_flag = 1
    opt.save_dir = opt.save_dir + '/' + opt.datasetname
    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    opt.save_dir = opt.save_dir + '/' + opt.model_name
    if (not os.path.exists(opt.save_dir)):
        os.mkdir(opt.save_dir)
    opt.save_results_dir = opt.save_dir+'/results'
    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/DSFNet.pth'

    print(modelPath)

    results_name = opt.model_name+'_'+modelPath.split('/')[-2]+'_'+modelPath.split('/')[-1].split('.')[0]

    test(opt, split, modelPath, show_flag, results_name, save_flag)