import numpy as np
import xml.dom.minidom as doxml

class eval_metric():
    def __init__(self, dis_th = 0.5, iou_th=0.05, eval_mode = 'dis'):
        self.dis_th = dis_th
        self.iou_th = iou_th
        self.eval_mode = eval_mode
        self.area_min_th = 2
        self.area_max_th = 80
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0


    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def get_result(self):
        precision = self.tp/(self.tp + self.fp+1e-7)
        recall = self.tp/(self.tp + self.fn+1e-7)
        f1 = 2*recall*precision/(recall+precision+1e-7)

        out = {}
        out['recall'] = recall*100
        out['prec'] = precision*100
        out['f1'] = f1*100
        out['tp'] = self.tp
        out['fp'] = self.fp
        out['fn'] = self.fn

        return out

    def update(self, gt, det):
        if (gt.shape[0] > 0):
            if (det.shape[0] > 0):
                if self.eval_mode == 'iou':
                    cost_matrix = self.iou_batch(det, gt)
                elif self.eval_mode == 'dis':
                    cost_matrix = self.dist_batch(det, gt)
                    if min(cost_matrix.shape) > 0:
                        cost_matrix[cost_matrix > self.dis_th] = self.dis_th + 10
                else:
                    raise Exception('Not a valid eval mode!!!!')

                if min(cost_matrix.shape) > 0:
                    # matched_indices = self.linear_assignment(cost_matrix)
                    # matched_matrix = cost_matrix[matched_indices[:, 0], matched_indices[:, 1]]
                    if self.eval_mode == 'iou':
                        matched_indices = self.linear_assignment(-cost_matrix)
                        matched_matrix = cost_matrix[matched_indices[:, 0], matched_indices[:, 1]]
                        matched_results = matched_matrix[matched_matrix > self.iou_th]
                    elif self.eval_mode == 'dis':
                        matched_indices = self.linear_assignment(cost_matrix)
                        matched_matrix = cost_matrix[matched_indices[:, 0], matched_indices[:, 1]]
                        matched_results = matched_matrix[matched_matrix < self.dis_th]
                    else:
                        raise Exception('Not a valid eval mode!!!!')
                else:
                    matched_results = np.empty(shape=(0, 1))

                tp = matched_results.shape[0]
                fn = gt.shape[0] - tp
                fp = det.shape[0] - tp
            else:
                tp = 0
                fn = gt.shape[0]
                fp = 0
        else:
            tp = 0
            fn = 0
            fp = det.shape[0]

        self.tp += tp
        self.fn += fn
        self.fp += fp

    def iou_batch(self, bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
                  + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh + 1e-7)
        return (o)

    def dist_batch(self, bb_test, bb_gt):
        """
        From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        gt_center = (bb_gt[:, :, :2] + bb_gt[:, :, 2:4]) / 2
        det_center = (bb_test[:, :, :2] + bb_test[:, :, 2:4]) / 2
        o = np.sqrt(np.sum((gt_center - det_center) ** 2, -1))
        return (o)

    def linear_assignment(self, cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])  #
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))

    def getGtFromXml(self, xml_file):
        # tree = ET.parse(xml_file)
        tree = doxml.parse(xml_file)
        # root = tree.getroot()
        annotation = tree.documentElement

        objectlist = annotation.getElementsByTagName('object')

        gt = []

        if (len(objectlist) > 0):
            for object in objectlist:
                bndbox = object.getElementsByTagName('bndbox')
                for box in bndbox:
                    xmin0 = box.getElementsByTagName('xmin')
                    xmin = int(xmin0[0].childNodes[0].data)
                    ymin0 = box.getElementsByTagName('ymin')
                    ymin = int(ymin0[0].childNodes[0].data)
                    xmax0 = box.getElementsByTagName('xmax')
                    xmax = int(xmax0[0].childNodes[0].data)
                    ymax0 = box.getElementsByTagName('ymax')
                    ymax = int(ymax0[0].childNodes[0].data)
                    gt.append([xmin, ymin, xmax, ymax])
        return np.array(gt)