import numpy as np
from opts import opt
import torch.nn.functional as F
from scipy.spatial.distance import cdist


# Resize a rectangular image to a padded rectangular
def letterbox(img, color=0):
    # shape = [height, width]
    shape = img.shape[1:]
    ratio = min(float(opt.patch_size[0]) / shape[0], float(opt.patch_size[1]) / shape[1])

    # new_shape = [channel, height, width]
    new_shape = (round(shape[0] * ratio), round(shape[1] * ratio))

    # Padding
    dh = (opt.patch_size[0] - new_shape[0]) / 2
    dw = (opt.patch_size[1] - new_shape[1]) / 2

    # Top, bottom, left, right
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border, padded rectangular
    img = F.interpolate(img.unsqueeze(0), size=new_shape, mode='area')
    img = F.pad(img, (left, right, top, bottom, 0, 0, 0, 0), mode='constant', value=color)

    return img


def cxcywh2xywh(cxcywh):
    return [cxcywh[0] - cxcywh[2] / 2, cxcywh[1] - cxcywh[3] / 2, cxcywh[2], cxcywh[3]]


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(np.zeros(all_pairs_max_ymin.shape), all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(np.zeros(all_pairs_max_xmin.shape), all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths


def calc_iou(boxes1, boxes2):
    boxes1, boxes2 = xywh2xyxy(boxes1), xywh2xyxy(boxes2)
    intersect = intersection(boxes1, boxes2)
    area1, area2 = area(boxes1), area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def calc_ioa(boxes1, boxes2):
    boxes1, boxes2 = xywh2xyxy(boxes1), xywh2xyxy(boxes2)
    intersect = intersection(boxes1, boxes2)

    # Intersection area over box1's area
    areas = np.expand_dims(area(boxes1), axis=1)
    ioa_1 = intersect / areas

    # Intersection area over box2's area
    areas = np.expand_dims(area(boxes2), axis=0)
    ioa_2 = intersect / areas

    return np.maximum(ioa_1, ioa_2)


def class_agnostic_nms(tracks):
    while True:
        # Decode
        boxes = np.concatenate([np.array(track.tlwh)[None, :] for track in tracks], axis=0)

        # Check connectivity
        ioa = calc_ioa(boxes, boxes)
        con = 0.25 <= ioa

        # End condition
        if np.sum(con) == len(boxes):
            break

        # Pick idx to delete
        del_idx_can = np.where(np.sum(con, axis=0) >= 2)[0]
        scores = [tracks[d_idx_c].x1y1x2y2[3] for d_idx_c in del_idx_can]
        del_idx = del_idx_can[scores.index(min(scores))]
        tracks.pop(del_idx)

    return tracks


def class_agnostic_nms_det(detection, feat):
    while True:
        # Decode
        boxes = detection[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2

        # Check connectivity
        ioa = calc_ioa(boxes, boxes)
        con = 0.25 <= ioa

        # End condition
        if np.sum(con) == len(boxes):
            break

        # Pick idx to delete
        del_idx_can = np.where(np.sum(con, axis=0) >= 2)[0]
        scores = [boxes[d_idx_c, 1] + boxes[d_idx_c, 3] for d_idx_c in del_idx_can]
        detection = np.delete(detection, del_idx_can[scores.index(min(scores))], 0)
        feat = np.delete(feat, del_idx_can[scores.index(min(scores))], 0)

    return detection, feat


def pairwise_tracks_dist(clusters_dict, tracks, fdx, metric):
    dists = np.ones((len(clusters_dict), len(tracks)))

    for row, global_id in enumerate(clusters_dict.keys()):
        for col, track in enumerate(tracks):
            # To make them connect only with the lost clusters (For no MTSC erase)
            if fdx - clusters_dict[global_id].end_frame == 0:
                continue

            # Should not appear in same camera
            if track.cam in clusters_dict[global_id].cam_list:
                continue

            # Calculate distance
            dists[row, col] = np.min(cdist(clusters_dict[global_id].get_feature(),
                                           track.get_feature(mode=opt.get_feat_mode)[np.newaxis, :], metric))

    dists = np.clip(dists, 0, 1) if metric == 'cosine' else dists

    return dists


def pairwise_clusters_dist_naive(cluster_a, cluster_b_feat, metric):
    dist = np.ones((len(cluster_a.keys()), len(cluster_b_feat.keys())))

    for row, global_id in enumerate(cluster_a.keys()):
        for col, cdx in enumerate(cluster_b_feat.keys()):
            dist[row, col] = np.min(cdist(cluster_a[global_id].get_feature, cluster_b_feat[cdx], metric))

    dist = np.clip(dist, 0, 1) if metric == 'cosine' else dist

    return dist
