from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import torch

from lib.utils.opts import opts

from lib.models.stNet import get_det_net, load_model
from lib.dataset.coco import COCO

from lib.external.nms import soft_nms

from lib.utils.decode import ctdet_decode
from lib.utils.post_process import ctdet_post_process

from lib.utils.sort import *

import cv2

from progress.bar import Bar

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

def test(opt, split, modelPath, show_flag, results_name):

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # Logger(opt)
    print(opt.model_name)

    dataset = COCO(opt, split)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = get_det_net({'hm': dataset.num_classes, 'wh': 2, 'reg': 2}, opt.model_name)  # 建立模型
    model = load_model(model, modelPath)
    model = model.cuda()
    model.eval()

    results = {}
    return_time = False
    scale = 1
    num_classes = dataset.num_classes
    max_per_image = opt.K

    file_folder_pre = ''
    im_count = 0

    saveTxt = opt.save_track_results
    if saveTxt:
        track_results_save_dir = os.path.join(opt.save_results_dir, 'trackingResults'+opt.model_name)
        if not os.path.exists(track_results_save_dir):
            os.mkdir(track_results_save_dir)

    num_iters = len(data_loader)
    bar = Bar('processing', max=num_iters)
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # print(ind)
        if(ind>len(data_loader)-1):
            break

        bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters,total=bar.elapsed_td, eta=bar.eta_td
        )

        #set tracker
        file_folder_cur = pre_processed_images['file_name'][0].split('/')[-3]
        if file_folder_cur != file_folder_pre:
            if saveTxt and file_folder_pre!='':
                 fid.close()
            file_folder_pre = file_folder_cur
            mot_tracker = Sort()
            if saveTxt:
                im_count = 0
                txt_path = os.path.join(track_results_save_dir, file_folder_cur+'.txt')
                fid = open(txt_path, 'w+')

        #read images
        detection = []
        meta = pre_process(pre_processed_images['input'], scale)
        image = pre_processed_images['input'].cuda()
        img = pre_processed_images['imgOri'].squeeze().numpy()

        #detection
        output, dets = process(model, image, return_time)
        #POST PROCESS
        dets = post_process(dets, meta, num_classes)
        detection.append(dets)
        ret = merge_outputs(detection, num_classes, max_per_image)

        #update tracker
        dets_track = dets[1]
        dets_track_select = np.argwhere(dets_track[:,-1]>CONFIDENCE_thres)
        dets_track = dets_track[dets_track_select[:,0],:]
        track_bbs_ids = mot_tracker.update(dets_track)

        if(show_flag):
            frame, det = cv2_demo(img, track_bbs_ids)
            cv2.imshow('frame',frame)
            cv2.waitKey(5)
            hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
            cv2.imshow('hm', hm1)
            cv2.waitKey(5)

        if saveTxt:
            im_count += 1
            track_bbs_ids = track_bbs_ids[::-1,:]
            track_bbs_ids[:,2:4] = track_bbs_ids[:,2:4]-track_bbs_ids[:,:2]
            for it in range(track_bbs_ids.shape[0]):
                fid.write('%d,%d,%0.2f,%0.2f,%0.2f,%0.2f,1,-1,-1,-1\n'%(im_count,
                          track_bbs_ids[it,-1], track_bbs_ids[it,0],track_bbs_ids[it,1],
                                track_bbs_ids[it, 2], track_bbs_ids[it, 3]))

        results[img_id.numpy().astype(np.int32)[0]] = ret
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_results_dir, results_name)

if __name__ == '__main__':
    opt = opts().parse()

    split = 'test'
    show_flag = opt.save_track_results
    if (not os.path.exists(opt.save_results_dir)):
        os.mkdir(opt.save_results_dir)

    if opt.load_model != '':
        modelPath = opt.load_model
    else:
        modelPath = './checkpoints/DSFNet.pth'
    print(modelPath)

    results_name = opt.model_name+'_'+modelPath.split('/')[-1].split('.')[0]
    test(opt, split, modelPath, show_flag, results_name)