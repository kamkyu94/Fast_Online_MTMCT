import lap
import numpy as np
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(a_x1y1x2y2s, b_x1y1x2y2s):
    ious = np.zeros((len(a_x1y1x2y2s), len(b_x1y1x2y2s)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(np.ascontiguousarray(a_x1y1x2y2s, dtype=np.float),
                     np.ascontiguousarray(b_x1y1x2y2s, dtype=np.float))

    return ious


def iou_distance(a_tracks, b_tracks):
    a_x1y1x2y2s = [track.x1y1x2y2 for track in a_tracks]
    b_x1y1x2y2s = [track.x1y1x2y2 for track in b_tracks]

    _ious = ious(a_x1y1x2y2s, b_x1y1x2y2s)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)

    # Normalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix
