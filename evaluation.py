import numpy as np
import scipy.io as sio
import os
from lib.utils.utils_eval import eval_metric

if __name__ == '__main__':
    #eval func
    eval_mode_metric = 'iou'
    dis_th = [5]
    iou_th = [0.05, 0.1, 0.2]
    conf_thresh = 0.3
    dataName = [3,5,2,8,10,6,9]
    #data path ori
    ANN_PATH0 = './dataset/RsCarData/images/test1024/'
    #specify results path
    results_dir_tol = [
        './weights/rsdata/DSFNet/results/DSFNet_checkpoints_DSFNet_mat/',
    ]

    methods_results = {}
    for results_dir0 in results_dir_tol:
        iou_results = []
        print(results_dir0)
        #record the results
        txt_name = 'reuslts_%s_%.2f.txt'%(eval_mode_metric, conf_thresh)
        fid = open(results_dir0 + txt_name, 'w+')
        fid.write(results_dir0 + '(recall,precision,F1)\n')
        fid.write(eval_mode_metric + '\n')
        if eval_mode_metric=='dis':
            thres = dis_th
        elif eval_mode_metric=='iou':
            thres = iou_th
        else:
            raise Exception('Not a valid eval mode!!')
        ##eval
        thresh_results = {}
        for thre in thres:#
            if eval_mode_metric == 'dis':
                dis_th_cur = thre
                iou_th_cur = 0.05
            elif eval_mode_metric == 'iou':
                dis_th_cur = 5
                iou_th_cur = thre
            else:
                raise Exception('Not a valid eval mode!!')
            det_metric = eval_metric(dis_th=dis_th_cur, iou_th=iou_th_cur, eval_mode=eval_mode_metric)
            fid.write('conf_thresh=%.2f,thresh=%.2f\n'%(conf_thresh, thre))
            print('conf_thresh=%.2f,thresh=%.2f'%(conf_thresh, thre))
            results_temp = {}
            for datafolder in dataName:
                det_metric.reset()
                ANN_PATH = ANN_PATH0+'%03d'%datafolder+'/xml_det/'
                results_dir = results_dir0 + '%03d/' % (datafolder)
                #start eval
                anno_dir = os.listdir(ANN_PATH)
                num_images = len(anno_dir)
                for index in range(num_images):
                    file_name = anno_dir[index]
                    #load gt
                    if(not file_name.endswith('.xml')):
                        continue
                    annName = ANN_PATH+file_name
                    if not os.path.exists(annName):
                        continue
                    gt_t = det_metric.getGtFromXml(annName)
                    #load det
                    matname = results_dir + file_name.replace('.xml','.mat')
                    if os.path.exists(matname):
                        det_ori = sio.loadmat(matname)['A']
                        det = np.array(det_ori)
                        score = det[:,-1]
                        inds = np.argsort(-score)
                        score = score[inds]
                        det = det[score>conf_thresh]
                    else:
                        det = np.empty([0,4])
                    #eval
                    det_metric.update(gt_t, det)
                #get results
                result = det_metric.get_result()
                fid.write('&%.2f\t&%.2f\t&%.2f\n' % (result['recall'], result['prec'], result['f1']))
                print('%s, evalmode=%s, thre=%0.2f, conf_th=%0.2f, re=%0.3f, prec=%0.3f, f1=%0.3f' % (
                '%03d' % datafolder, eval_mode_metric, thre, conf_thresh, result['recall'], result['prec'], result['f1']))
                results_temp[datafolder] = result
            #avg results
            meatri = [[v['recall'], v['prec'], v['f1']] for k, v in results_temp.items()]
            meatri = np.array(meatri)
            avg_results = np.mean(meatri, 0)
            print('avg result:  ', avg_results)
            fid.write(
                '&%.2f\t&%.2f\t&%.2f\n' % (avg_results[0], avg_results[1], avg_results[2]))
            results_temp['avg'] = {
                'recall': avg_results[0],
                'prec': avg_results[1],
                'f1': avg_results[2],
            }
            thresh_results[thre] = results_temp
        methods_results[results_dir0] = thresh_results
    # print(methods_results)