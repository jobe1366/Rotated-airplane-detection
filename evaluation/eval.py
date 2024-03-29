"""This script is used to get mAP result on val set."""
import numpy as np
import matplotlib.pyplot as plt
from polyiou import polyiou
import os

filepath = os.path.dirname(__file__)


def parse_gt(filename):

    objects = []
    try:
        with open( filename , 'r') as f:
            while True:
                line = f.readline()
                if line:
                    splitlines = line.strip().split(' ')
                    object_struct = {}
                    if (len(splitlines) < 9):
                        continue
                    object_struct['name'] = splitlines[8]
                    object_struct['difficult'] = 0
                    object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                    objects.append(object_struct)
                else:
                    break
    except:
        f'{filename} not exists'
    return objects   


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(os.path.join(filepath, annopath.format(imagename))) 
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    print('imge_ids len:', len(image_ids))
    nd = len(image_ids) 
    tp = np.zeros(nd).astype(np.int8)
    fp = np.zeros(nd).astype(np.int8)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(np.float64)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(np.float64)
        if BBGT.size > 0:
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)  

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                jmax = BBGT_keep_index[jmax]  
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.0

    print('check fp:', fp)
    print('check tp', tp)
    print('npos num:', npos)
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    print('check fp:', fp)
    print('check tp', tp)

    rec = tp /  float(npos)                     
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)  
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def main(detpath, annopath, imagesetfile, classnames):
    """
    Args:
        detpath: detection results for each class
        annopath: GT label for each image file
        imagesetfile: the image lists
    """

    classaps = []

    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)
        plt.figure(figsize=(8, 4))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec)
        plt.savefig('./airplane_detection_PRCURVE.jpg')
        plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)


if __name__ == '__main__':
    main(
        detpath =   'evaluation/result_classname/Task1_{:s}.txt',
         annopath = 'gt_labels/{:s}.txt',
         imagesetfile = 'evaluation/imgnamefile.txt',
         classnames=  ['plane']  
         )
